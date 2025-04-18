# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import Callable, Iterator, List, Optional, Union

import torch
from torch.autograd.variable import Variable
import torch.distributed
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron.core import parallel_state
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.enums import ModelType
from megatron.core.utils import get_attr_wrapped_model, get_model_type
import time
import json
from megatron import get_args

# Types
Shape = Union[List[int], torch.Size]

def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of
        modules in the case of interleaved pipeline parallelism.

    num_microbatches (int, required):
        The number of microbatches to go through

    dtype (required when using pipeline parallelism): dtype used in
        p2p communication, usually params_dtype

    tensor_shape (required when using pipeline parallelism): Shape of
        tensor. The tensor is expected to be 3D and its order of
        dimension is supposed to be ``(sequence, batch, hidden)``.

    decoder_seq_length (int, required for ModelType.encoder_and_decoder models):
        Sequence length of the decoder portion, used to determine tensor shapes.

    grad_scaler (optional, default=None): If using loss scaling,
        this function should take the loss and return the scaled
        loss. If None, no function is called on the loss.

    sequence_parallel (optional, default=False):
        Set to :obj:`True` for this function to handle sequence
        length.  When :obj:`True`, the sequence length on each tensor
        model parallel rank is updated to
        :math:`original\_sequence\_length /
        tensor\_model\_parallel\_world\_size`.
        TODO: Do we need this? Just roll into tensor_shape arg?

    overlap_p2p_comm (optional, default=False): When True
        some of the peer to peer communication for pipeline
        parallelism will overlap with computation. Must be False if
        batch_p2p_comm is true.

    batch_p2p_comm (optional, default=True): When true use
        batch_isend_irecv, otherwise use individual isend and irecv
        calls. Must be false if overlap_p2p_comm is True.

    forward_only (optional, default=False): Perform only the forward step

    timers (optional, default=None): TODO

    collect_non_loss_data: TODO

    enable_autocast (optional, default=False): If True, runs the
        forward_step_func call inside torch.autocast context

    deallocate_pipeline_outputs (optional, default=False): If True, output data
        is deallocated after the tensor is sent to the next pipeline stage.
        Helps with saving memory, does nothing when pipeline parallel is
        not used.

    no_sync_func (optional): Function that creates a context that
        suppresses asynchronous data-parallel communication. If the
        model is an instance of torch.nn.DistributedDataParallel, the
        default is to use torch.nn.DistributedDataParallel.no_sync.

    grad_sync_func (optional): Function that launches asynchronous
        gradient reductions (e.g. distributed optimizer gradient
        reduce-scatters). The function should take one argument: an
        iterable of parameters whose gradients are to be synchronized.

    param_sync_func (optional): Function that launches asynchronous
        parameter synchronizations (e.g. distributed optimizer
        parameter all-gathers). The function should take one argument:
        an iterable of parameters to be synchronized.

    """
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        if not parallel_state.get_pipeline_is_1f1b_enable():
            if torch.distributed.get_rank() == 0:
                print('\ngpipe\n')
            forward_backward_func = forward_backward_pipelining_gpipe
        elif parallel_state.get_wpp_model_parallel_world_size() is not None:
            if torch.distributed.get_rank() == 0:
                print('\nwpp\n')
            forward_backward_func = forward_backward_pipelining_with_wpp
        elif parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            if torch.distributed.get_rank() == 0:
                print('\ninterleaving PP\n')
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            if torch.distributed.get_rank() == 0:
                print('\nuninterleaving PP\n')
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func

def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), \
        "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, \
        "counter-productive to free a view of another tensor."
    out.data = torch.empty(
        (1,),
        device = out.device,
        dtype = out.dtype,
    )

def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, \
        "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), \
        "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), \
        "grad_output == '%s'." % type(grad_output).__name__

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(
            output,
            memory_format = torch.preserve_format,
        )

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors = (output,),
        grad_tensors = (grad_output,),
        keep_graph = False,
        create_graph = False,
        inputs = tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )





def forward_step(forward_step_func,
                 data_iterator,
                 model,
                 num_microbatches,
                 input_tensor,
                 forward_data_store,
                 timers,
                 collect_non_loss_data=False,
                 autocast_dtype=torch.float,
                 enable_autocast=False):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    if timers is not None:
        timers('forward-compute', log_level=2).start()

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if enable_autocast:
        context_manager = torch.autocast("cuda", dtype=autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        output_tensor, loss_func = forward_step_func(data_iterator, model)

    if parallel_state.is_pipeline_last_stage():
        if not collect_non_loss_data:
            output_tensor = loss_func(output_tensor)
            loss, loss_reduced = output_tensor
            output_tensor = loss / num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if timers is not None:
        timers('forward-compute').stop()
    
    # args = get_args()
    # rank = torch.distributed.get_rank()
    # log_file_name = args.save+'/log%d.txt'%rank
    # with open(log_file_name, 'a') as log_file:  
    #     log_file.write('---\n')
    #     log_file.write("model size%s\n"%str(parallel_state.get_virtual_pipeline_model_parallel_world_size()))
    #     log_file.write("model rank%s\n"%str(parallel_state.get_virtual_pipeline_model_parallel_rank()))
    #     log_file.write("pipeline rank%s\n"%str(parallel_state.get_pipeline_model_parallel_rank()))
    #     if parallel_state.is_pipeline_last_stage():
    #         log_file.write('last stage')
    #     log_file.write(str(output_tensor))
    #     log_file.write('---\n')

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)

    if parallel_state.is_pipeline_stage_after_split() and \
            model_type == ModelType.encoder_and_decoder:
        print(torch.distributed.get_rank(), 'is after split')
        return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor   
    return [output_tensor]


def backward_step(grad_scaler, input_tensor, output_tensor,
                  output_tensor_grad, model_type, timers, deallocate_pipeline_outputs=False):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.
    
    rank = torch.distributed.get_rank()
    args = get_args()
    log_file_name = args.save+'/log%d.txt'%rank

    if timers is not None:
        timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and grad_scaler is not None:
        output_tensor[0] = grad_scaler(output_tensor[0])
        
    # with open(log_file_name, 'a') as log_file:
    #     log_file.write(f"""input_tensor: {input_tensor}\n""")
    #     log_file.write(f"""@output_tensor: {output_tensor}\n""")
    #     log_file.write(f"""@output_tensor_grad: {output_tensor_grad}\n""")
    #     log_file.write(f"""input_tensor_grad: {input_tensor_grad}\n\n""")

    if deallocate_pipeline_outputs:
        custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        # , retain_graph=True
        if rank == 4:
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0], retain_graph=True)
        else:
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if parallel_state.get_pipeline_model_parallel_world_size() > 1 and \
            parallel_state.is_pipeline_stage_after_split() and \
            model_type == ModelType.encoder_and_decoder:
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    # with open(log_file_name, 'a') as log_file:
    #     log_file.write(f"""input_tensor: {input_tensor}\n""")
    #     log_file.write(f"""@output_tensor: {output_tensor}\n""")
    #     log_file.write(f"""@output_tensor_grad: {output_tensor_grad}\n""")
    #     log_file.write(f"""input_tensor_grad: {input_tensor_grad}\n\n""")

    if timers is not None:
        timers('backward-compute').stop()

    return input_tensor_grad


def forward_backward_no_pipelining(*,
                                   forward_step_func,
                                   data_iterator: Union[Iterator, List[Iterator]],
                                   model: Union[torch.nn.Module, List[torch.nn.Module]],
                                   num_microbatches: int,
                                   dtype: Optional[torch.dtype] = None,
                                   tensor_shape: Optional[Shape] = None, # unused
                                   decoder_seq_length: Optional[int] = None, # unused
                                   grad_scaler: Callable = None,
                                   sequence_parallel: bool = False, # unused
                                   overlap_p2p_comm: bool = False, # unused
                                   batch_p2p_comm: bool = True, # unused
                                   forward_only: bool = False,
                                   timers: Callable = None,
                                   collect_non_loss_data: bool = False,
                                   enable_autocast: bool = False,
                                   deallocate_pipeline_outputs: bool = False,
                                   no_sync_func: Optional[Callable] = None,
                                   grad_sync_func: Optional[Callable] = None, # unused
                                   param_sync_func: Optional[Callable] = None, # unused
                                   get_dependency: bool = False,
                                   ):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):
        assert len(model) == 1, \
            "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, \
            "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    if no_sync_func is None and isinstance(model, torchDDP):
        no_sync_func = model.no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)
    
    memory = [(time.time(), torch.cuda.memory_allocated())]

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor = forward_step(forward_step_func, data_iterator,
                                         model, num_microbatches, input_tensor, forward_data_store,
                                         timers, collect_non_loss_data, dtype, enable_autocast)
            memory.append((time.time(), torch.cuda.memory_allocated()))
            if not forward_only:
                backward_step(grad_scaler, input_tensor, output_tensor,
                              output_tensor_grad, model_type, timers, deallocate_pipeline_outputs)
            memory.append((time.time(), torch.cuda.memory_allocated()))

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = forward_step(forward_step_func, data_iterator,
                                 model, num_microbatches, input_tensor, forward_data_store,
                                 timers, collect_non_loss_data, dtype, enable_autocast)

    memory.append((time.time(), torch.cuda.memory_allocated()))
    if not forward_only:
        backward_step(grad_scaler, input_tensor, output_tensor,
                      output_tensor_grad, model_type, timers, deallocate_pipeline_outputs)
    memory.append((time.time(), torch.cuda.memory_allocated()))
    
    args = get_args()
    record_path = args.save+'/memorylog%d.json'%torch.distributed.get_rank()
    with open(record_path, 'r') as file:
        data = json.load(file)
    data['memory'].extend(memory)
    with open(record_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return forward_data_store, {}


def forward_backward_pipelining_with_interleaving(*,
                                                  forward_step_func,
                                                  data_iterator: Union[Iterator, List[Iterator]],
                                                  model: Union[torch.nn.Module, List[torch.nn.Module]],
                                                  num_microbatches: int,
                                                  dtype: torch.dtype,
                                                  tensor_shape: Shape,
                                                  decoder_seq_length: Optional[int] = None,
                                                  grad_scaler: Callable = None,
                                                  sequence_parallel: bool = False,
                                                  overlap_p2p_comm: bool = False,
                                                  batch_p2p_comm: bool = True,
                                                  forward_only: bool = False,
                                                  timers: Callable = None,
                                                  collect_non_loss_data: bool = False,
                                                  enable_autocast: bool = False,
                                                  deallocate_pipeline_outputs: bool = False,
                                                  no_sync_func: Optional[Callable] = None,
                                                  grad_sync_func: Optional[Callable] = None,
                                                  param_sync_func: Optional[Callable] = None,
                                                  get_dependency: bool = False,
                                                  ):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    assert isinstance(model, list), \
        "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), \
        "invalid model chunking"
    assert isinstance(data_iterator, list), \
        "interleaved pipeline parallelism expected each model chunk to have a data iterator"

    if overlap_p2p_comm and batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Disable async grad reductions
    if no_sync_func is None and all(isinstance(chunk, torchDDP) for chunk in model):
        def multi_no_sync():
            stack = contextlib.ExitStack()
            for chunk in model:
                stack.enter_context(chunk.no_sync())
            return stack
        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None
    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()
    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None
    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != tensor_shape[0]:
        raise RuntimeError("Interleaving is not supported with a different decoder sequence length.")

    if sequence_parallel:
        seq_length, batch_size, hidden = tensor_shape
        tensor_shape = (
            seq_length // parallel_state.get_tensor_model_parallel_world_size(),
            batch_size,
            hidden,
        )

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    all_warmup_microbatches = False
    # if torch.distributed.get_rank() == 0:
    #     # print(model)
    #     print('total_num_microbatches', total_num_microbatches)
    #     print('num_microbatches', num_microbatches)
    #     print('pipeline_parallel_size', pipeline_parallel_size)
    #     print('num_model_chunks', num_model_chunks)
    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        
        # TODO：不是很明白为什么要这么做
        if num_microbatches == pipeline_parallel_size:
            num_warmup_microbatches = total_num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = \
                (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (
                num_model_chunks - 1) * pipeline_parallel_size
        num_warmup_microbatches = min(num_warmup_microbatches,
                                        total_num_microbatches)
    num_microbatches_remaining = \
        total_num_microbatches - num_warmup_microbatches

    # if torch.distributed.get_rank() == 0:
    # print(torch.distributed.get_rank(), 'num_warmup_microbatches', num_warmup_microbatches)
    # Synchronize params for first two model chunks
    if param_sync_func is not None:
        param_sync_func(model[0].parameters())
        param_sync_func(model[1].parameters())

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = (num_model_chunks - model_chunk_id - 1)
        return model_chunk_id

    def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == 0:
            return microbatch_id_in_group % pipeline_parallel_size == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == num_microbatch_groups - 1:
            return microbatch_id_in_group % pipeline_parallel_size == pipeline_parallel_size - 1
        else:
            return False


    def forward_step_helper(microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        # print('forward_step_helper, stage id:', torch.distributed.get_rank(),
        #       ', microbatch id:', microbatch_id)
        rank = torch.distributed.get_rank()
        args = get_args()
        # log_file_name = args.save+'/log%d.txt'%rank
        # with open(log_file_name, 'a') as log_file:
        #     log_file.write(f"""forward_step_helper, time:{time.time()},
        #                    microbatch id: {microbatch_id}\n""")
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        # 是在同步吗？这步是必须的吗？
        # if param_sync_func is not None:
        #     param_sync_microbatch_id = microbatch_id + pipeline_parallel_rank
        #     if param_sync_microbatch_id < num_microbatches and is_first_microbatch_for_model_chunk(param_sync_microbatch_id):
        #         param_sync_chunk_id = get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
        #         if 1 < param_sync_chunk_id < num_model_chunks:
        #             param_sync_func(model[param_sync_chunk_id].parameters())

        # forward step
        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == \
                    len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(forward_step_func,
                                     data_iterator[model_chunk_id],
                                     model[model_chunk_id],
                                     num_microbatches,
                                     input_tensor,
                                     forward_data_store,
                                     timers,
                                     collect_non_loss_data,
                                     dtype,
                                     enable_autocast)
        output_tensors[model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        # print('backward_step_helper, stage id:', torch.distributed.get_rank(),
        #       ', microbatch id:', microbatch_id)
        rank = torch.distributed.get_rank()
        args = get_args()
        # log_file_name = args.save+'/log--%d.txt'%rank
        # with open(log_file_name, 'a') as log_file:
        #     log_file.write(f"""backward_step_helper, time:{time.time()}, 
        #                    microbatch id: {microbatch_id}\n""")
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch grad synchronization (default)
        # 这是在做什么？也是同步吗？是必须的吗？
        # if grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
        #     enable_grad_sync()
        #     synchronized_model_chunks.add(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = \
            backward_step(grad_scaler,
                          input_tensor,
                          output_tensor,
                          output_tensor_grad,
                          model_type,
                          timers,
                          deallocate_pipeline_outputs)
        # with open(log_file_name, 'a') as log_file:
        #     log_file.write(f"""model_chunk_id: {model_chunk_id}\n""")
        #     # log_file.write(f"""input_tensor: {input_tensor}\n""")
        #     # log_file.write(f"""output_tensor: {output_tensor}\n""")
        #     log_file.write(f"""output_tensor_grad: {output_tensor_grad}\n""")
        #     log_file.write(f"""input_tensor_grad: {input_tensor_grad}\n""")
        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if grad_sync_func is not None:
            grad_sync_microbatch_id = microbatch_id - pipeline_parallel_rank
            if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(grad_sync_microbatch_id):
                grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                enable_grad_sync()
                grad_sync_func(model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

        return input_tensor_grad
    
    memory = []
    memory.append((time.time(), torch.cuda.memory_allocated()))

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(
        p2p_communication.recv_forward(tensor_shape,
                                       dtype=dtype,
                                       batch_p2p_comm=batch_p2p_comm,
                                       timers=timers))

    fwd_wait_handles = None
    bwd_wait_handles = None
    
    memory.append((time.time(), torch.cuda.memory_allocated()))

    for k in range(num_warmup_microbatches):

        if fwd_wait_handles is not None:
            for req in fwd_wait_handles:
                req.wait()

        output_tensor = forward_step_helper(k)
        memory.append((time.time(), torch.cuda.memory_allocated()))

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k+1, forward=True)
        recv_prev = True
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not overlap_p2p_comm:
            # if torch.distributed.get_rank() == 0:
            #     print('not overlap_p2p_comm')
            if k == (num_warmup_microbatches - 1) and not forward_only and \
                    not all_warmup_microbatches:
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                input_tensor, output_tensor_grad = \
                    p2p_communication.send_forward_backward_recv_forward_backward(
                        output_tensor, input_tensor_grad,
                        recv_prev=recv_prev, recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        dtype=dtype,
                        batch_p2p_comm=batch_p2p_comm,
                        timers=timers)
                output_tensor_grads[num_model_chunks-1].append(output_tensor_grad)
            else:
                input_tensor = \
                    p2p_communication.send_forward_recv_forward(
                        output_tensor, recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        dtype=dtype,
                        batch_p2p_comm=batch_p2p_comm,
                        timers=timers)
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        else:
            input_tensor, fwd_wait_handles = \
                p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                    overlap_p2p_comm=True)

            if k == (num_warmup_microbatches - 1) and not forward_only and \
                    not all_warmup_microbatches:
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False

                output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    batch_p2p_comm=batch_p2p_comm,
                    dtype=dtype,
                    timers=timers,
                    overlap_p2p_comm=True)

                output_tensor_grads[num_model_chunks-1].append(output_tensor_grad)
            input_tensors[next_forward_model_chunk_id].append(input_tensor)

        deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        if overlap_p2p_comm:
            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    req.wait()

            deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)

            output_tensor = forward_step_helper(forward_k)
            memory.append((time.time(), torch.cuda.memory_allocated()))

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

            # Last virtual stage no activation tensor to send
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True)
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1,
                                                                forward=True)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Send activation tensor to the next stage and receive activation tensor from the
            # previous stage
            input_tensor, fwd_wait_handles = \
                p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers,
                    overlap_p2p_comm=True)
            # assert fwd_wait_handles is not None

            if bwd_wait_handles is not None:
                for req in bwd_wait_handles:
                    req.wait()

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)
            memory.append((time.time(), torch.cuda.memory_allocated()))

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)

            # First virtual stage no activation gradient tensor to send
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if the current virtual stage has an activation gradient tensor to receive
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k + 1, forward=False
                )

            output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                input_tensor_grad, recv_next=recv_next,
                tensor_shape=tensor_shape,
                dtype=dtype,
                batch_p2p_comm=batch_p2p_comm,
                timers=timers,
                overlap_p2p_comm=True)

        else: # no p2p overlap
            output_tensor = forward_step_helper(forward_k)
            memory.append((time.time(), torch.cuda.memory_allocated()))

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)
            memory.append((time.time(), torch.cuda.memory_allocated()))

            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True)
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1,
                                                                 forward=True)

            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False)
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1,
                                                                  forward=False)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            input_tensor, output_tensor_grad = \
                p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor, input_tensor_grad,
                    recv_prev=recv_prev, recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers)
            deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(
                output_tensor_grad)

    deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if overlap_p2p_comm and bwd_wait_handles is not None:
            for wait_handle in bwd_wait_handles:
                wait_handle.wait()

        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks-1].append(
                p2p_communication.recv_backward(tensor_shape,
                                                dtype=dtype,
                                                batch_p2p_comm=batch_p2p_comm,
                                                timers=timers))
        for k in range(num_microbatches_remaining, total_num_microbatches):
            input_tensor_grad = backward_step_helper(k)
            memory.append((time.time(), torch.cuda.memory_allocated()))
            next_backward_model_chunk_id = get_model_chunk_id(k+1, forward=False)
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (total_num_microbatches - 1):
                recv_next = False
            output_tensor_grads[next_backward_model_chunk_id].append(
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    dtype=dtype,
                    batch_p2p_comm=batch_p2p_comm,
                    timers=timers))

    # Launch any remaining grad reductions
    enable_grad_sync()
    if grad_sync_func is not None:
        params = []
        for model_chunk_id in range(num_model_chunks):
            if model_chunk_id not in synchronized_model_chunks:
                params.extend(model[model_chunk_id].parameters())
                synchronized_model_chunks.add(model_chunk_id)
        if params:
            grad_sync_func(params)
            
    memory.append((time.time(), torch.cuda.memory_allocated()))
    
    args = get_args()
    record_path = args.save+'/memorylog%d.json'%torch.distributed.get_rank()
    with open(record_path, 'r') as file:
        data = json.load(file)
    data['memory'].extend(memory)
    with open(record_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return forward_data_store, {}


def get_tensor_shapes(*,
                      rank: int,
                      model_type: ModelType,
                      tensor_shape: Shape,
                      decoder_seq_length: int,
                      sequence_parallel: bool):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    tensor_shapes = []

    assert (
        len(tensor_shape) == 3
    ), f"`tensor_shape` should be [sequence_length, micro_batch_size, hidden_size] but {tensor_shape}"

    seq_length, micro_batch_size, hidden_size = tensor_shape

    if sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()

    if model_type == ModelType.encoder_and_decoder:
        if sequence_parallel:
            decoder_seq_length = decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()

        if parallel_state.is_pipeline_stage_before_split(rank):
            tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
    else:
        tensor_shapes.append((seq_length, micro_batch_size, hidden_size))
    return tensor_shapes



def recv_forward(tensor_shapes, dtype, timers, batch_p2p_comm = True, from_prev=True):
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape, dtype,
                                                                timers=timers, batch_p2p_comm=batch_p2p_comm, 
                                                                from_prev=from_prev))
    return input_tensors


def recv_backward(tensor_shapes, dtype, timers, batch_p2p_comm = True, from_next=True):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape, dtype,
                                                                       timers=timers, batch_p2p_comm=batch_p2p_comm,
                                                                       from_next=from_next))
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, timers, batch_p2p_comm = True, to_next=True):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, timers=timers, 
                                       batch_p2p_comm=batch_p2p_comm, to_next=to_next)


def send_backward(input_tensor_grads, tensor_shapes, timers, batch_p2p_comm = True, to_prev=True):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, timers=timers, 
                                        batch_p2p_comm=batch_p2p_comm, to_prev=to_prev)


def send_forward_recv_backward(output_tensors, tensor_shapes, dtype, timers, 
                               batch_p2p_comm = True, with_next=True):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
                output_tensor, tensor_shape, dtype, timers=timers, 
                batch_p2p_comm=batch_p2p_comm, with_next=with_next)
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, dtype, timers, 
                               batch_p2p_comm = True, with_prev=True):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    # print('send_backward_recv_forward')
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        # if torch.distributed.get_rank() == 1:
        #     print(str(input_tensor_grad), tensor_shape)
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
                input_tensor_grad, tensor_shape, dtype, timers=timers, 
                batch_p2p_comm=batch_p2p_comm, with_prev=with_prev)
        input_tensors.append(input_tensor)
    return input_tensors


def forward_backward_pipelining_without_interleaving(*,
                                                     forward_step_func,
                                                     data_iterator: Union[Iterator, List[Iterator]],
                                                     model: Union[torch.nn.Module, List[torch.nn.Module]],
                                                     num_microbatches: int,
                                                     dtype: torch.dtype,
                                                     tensor_shape: Shape,
                                                     decoder_seq_length: Optional[int] = None,
                                                     grad_scaler: Callable = None,
                                                     sequence_parallel: bool = False,
                                                     overlap_p2p_comm: bool = False,
                                                     batch_p2p_comm: bool = True,
                                                     forward_only: bool = False,
                                                     timers: Callable = None,
                                                     collect_non_loss_data: bool = False,
                                                     enable_autocast: bool = False,
                                                     deallocate_pipeline_outputs: bool = False,
                                                     no_sync_func: Optional[Callable] = None,
                                                     grad_sync_func: Optional[Callable] = None,
                                                     param_sync_func: Optional[Callable] = None, # unused
                                                     get_dependency: bool = False,):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    args = get_args()

    if isinstance(model, list):
        assert len(model) == 1, \
            "non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, \
            "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    if overlap_p2p_comm:
        raise ValueError("Non-interleaved pipeline parallelism does not support overlapping p2p communication")

    if not batch_p2p_comm:
        raise ValueError("Non-interleaved pipeline parallelism only supports using batched p2p communication")

    # Disable async grad reductions
    if no_sync_func is None and isinstance(model, torchDDP):
        no_sync_func = model.no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None
    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()
    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None
    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = \
        (parallel_state.get_pipeline_model_parallel_world_size() -
         parallel_state.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches
        
    memory = []
    memory.append((time.time(), torch.cuda.memory_allocated()))

    model_type = get_model_type(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(rank=rank-1,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel)
    send_tensor_shapes = get_tensor_shapes(rank=rank,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel)

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []
    
    memory.append((time.time(), torch.cuda.memory_allocated()))

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        input_tensor = recv_forward(recv_tensor_shapes, dtype, timers=timers)
        output_tensor = forward_step(forward_step_func, data_iterator, model, num_microbatches,
                                     input_tensor, forward_data_store,
                                     timers, collect_non_loss_data, dtype, enable_autocast)
        memory.append((time.time(), torch.cuda.memory_allocated()))
        send_forward(output_tensor, send_tensor_shapes, timers=timers)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(recv_tensor_shapes, dtype, timers=timers)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = (i == (num_microbatches_remaining - 1))

        output_tensor = forward_step(forward_step_func, data_iterator, model, num_microbatches,
                                     input_tensor, forward_data_store,
                                     timers, collect_non_loss_data, dtype, enable_autocast)
        memory.append((time.time(), torch.cuda.memory_allocated()))
        # with open(args.save+'/log.txt', 'a') as log_file:
        #     if torch.distributed.get_rank() == 3:
        #         log_file.write(f"""output tensor %s\n""" % (output_tensor))

        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, timers=timers)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, dtype, timers=timers)

        else:
            output_tensor_grad = \
                send_forward_recv_backward(output_tensor,
                                           send_tensor_shapes, dtype,
                                           timers=timers)

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            input_tensor_grad = \
                backward_step(grad_scaler, input_tensor, output_tensor,
                              output_tensor_grad, model_type, timers, deallocate_pipeline_outputs)
            memory.append((time.time(), torch.cuda.memory_allocated()))

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, timers=timers)
            else:
                input_tensor = \
                    send_backward_recv_forward(
                        input_tensor_grad, recv_tensor_shapes, dtype, timers=timers)

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches-1:
                if grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, dtype, timers=timers)

            input_tensor_grad = \
                backward_step(grad_scaler, input_tensor, output_tensor,
                              output_tensor_grad, model_type, timers, deallocate_pipeline_outputs)
            memory.append((time.time(), torch.cuda.memory_allocated()))

            send_backward(input_tensor_grad, recv_tensor_shapes, timers=timers)

    # Launch any remaining grad reductions
    if no_sync_context is not None:
        enable_grad_sync()
        if grad_sync_func is not None:
            grad_sync_func(model.parameters())
            
    memory.append((time.time(), torch.cuda.memory_allocated()))
    
    args = get_args()
    record_path = args.save+'/memorylog%d.json'%torch.distributed.get_rank()
    with open(record_path, 'r') as file:
        data = json.load(file)
    data['memory'].extend(memory)
    with open(record_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return forward_data_store, {}

def forward_backward_pipelining_gpipe(*,
                                                     forward_step_func,
                                                     data_iterator: Union[Iterator, List[Iterator]],
                                                     model: Union[torch.nn.Module, List[torch.nn.Module]],
                                                     num_microbatches: int,
                                                     dtype: torch.dtype,
                                                     tensor_shape: Shape,
                                                     decoder_seq_length: Optional[int] = None,
                                                     grad_scaler: Callable = None,
                                                     sequence_parallel: bool = False,
                                                     overlap_p2p_comm: bool = False,
                                                     batch_p2p_comm: bool = True,
                                                     forward_only: bool = False,
                                                     timers: Callable = None,
                                                     collect_non_loss_data: bool = False,
                                                     enable_autocast: bool = False,
                                                     deallocate_pipeline_outputs: bool = False,
                                                     no_sync_func: Optional[Callable] = None,
                                                     grad_sync_func: Optional[Callable] = None,
                                                     param_sync_func: Optional[Callable] = None, # unused
                                                     get_dependency: bool = False,
                                                     ):
    """Run GPipe schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    
    # print("Run GPipe schedule ^_^")

    if isinstance(model, list):
        assert len(model) == 1, \
            "GPipe does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, \
            "GPipe schedule does not support model chunking"
        data_iterator = data_iterator[0]

    if overlap_p2p_comm:
        raise ValueError("GPipe does not support overlapping p2p communication")

    if not batch_p2p_comm:
        raise ValueError("GPipe only supports using batched p2p communication")

    # Disable async grad reductions
    if no_sync_func is None and isinstance(model, torchDDP):
        no_sync_func = model.no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None
    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()
    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None
    disable_grad_sync()
        
    memory = []
    memory.append((time.time(), torch.cuda.memory_allocated()))

    model_type = get_model_type(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(rank=rank-1,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel)
    send_tensor_shapes = get_tensor_shapes(rank=rank,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel)

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []
    
    memory.append((time.time(), torch.cuda.memory_allocated()))

    # Run forward passes.
    for i in range(num_microbatches):
        input_tensor = recv_forward(recv_tensor_shapes, dtype, timers=timers)
        output_tensor = forward_step(forward_step_func, data_iterator, model, num_microbatches,
                                     input_tensor, forward_data_store,
                                     timers, collect_non_loss_data, dtype, enable_autocast)
        memory.append((time.time(), torch.cuda.memory_allocated()))
        send_forward(output_tensor, send_tensor_shapes, timers=timers)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], deallocate_pipeline_outputs)

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_microbatches-1:
                if grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, dtype, timers=timers)

            input_tensor_grad = \
                backward_step(grad_scaler, input_tensor, output_tensor,
                              output_tensor_grad, model_type, timers, deallocate_pipeline_outputs)
            memory.append((time.time(), torch.cuda.memory_allocated()))

            send_backward(input_tensor_grad, recv_tensor_shapes, timers=timers)

    # Launch any remaining grad reductions
    if no_sync_context is not None:
        enable_grad_sync()
        if grad_sync_func is not None:
            grad_sync_func(model.parameters())
            
    memory.append((time.time(), torch.cuda.memory_allocated()))
    
    args = get_args()
    record_path = args.save+'/memorylog%d.json'%torch.distributed.get_rank()
    with open(record_path, 'r') as file:
        data = json.load(file)
    data['memory'].extend(memory)
    with open(record_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return forward_data_store, {}

def forward_backward_pipelining_with_wpp(*,
                                                  forward_step_func,
                                                  data_iterator: Union[Iterator, List[Iterator]],
                                                  model: Union[torch.nn.Module, List[torch.nn.Module]],
                                                  num_microbatches: int,
                                                  dtype: torch.dtype,
                                                  tensor_shape: Shape,
                                                  decoder_seq_length: Optional[int] = None,
                                                  grad_scaler: Callable = None,
                                                  sequence_parallel: bool = False,
                                                  overlap_p2p_comm: bool = False,
                                                  batch_p2p_comm: bool = True,
                                                  forward_only: bool = False,
                                                  timers: Callable = None,
                                                  collect_non_loss_data: bool = False,
                                                  enable_autocast: bool = False,
                                                  deallocate_pipeline_outputs: bool = False,
                                                  no_sync_func: Optional[Callable] = None,
                                                  grad_sync_func: Optional[Callable] = None,
                                                  param_sync_func: Optional[Callable] = None,
                                                  get_dependency: bool = False,
                                                  ):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    assert isinstance(model, list), \
        "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), \
        "invalid model chunking"
    assert isinstance(data_iterator, list), \
        "interleaved pipeline parallelism expected each model chunk to have a data iterator"

    if overlap_p2p_comm and batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Disable async grad reductions
    if no_sync_func is None and all(isinstance(chunk, torchDDP) for chunk in model):
        def multi_no_sync():
            stack = contextlib.ExitStack()
            for chunk in model:
                stack.enter_context(chunk.no_sync())
            return stack
        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None
    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()
    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None
    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    forward_data_store = []
    dependencies = {'forward':[], 'backward':[]}
   
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
    
    model_chunk_ids = {i:0 for i in range(num_microbatches)}

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != tensor_shape[0]:
        raise RuntimeError("Interleaving is not supported with a different decoder sequence length.")

    if sequence_parallel:
        seq_length, batch_size, hidden = tensor_shape
        tensor_shape = (
            seq_length // parallel_state.get_tensor_model_parallel_world_size(),
            batch_size,
            hidden,
        )

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    all_warmup_microbatches = False
    
    # if forward_only:
    #     # num_warmup_microbatches = total_num_microbatches
    #     # num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches
    #     num_forward_warmup_microbatches = total_num_microbatches
    #     num_forward_steady_microbatches = 0
    #     num_microbatches_remaining = 0
    # else:
    assert num_microbatches == pipeline_parallel_size, \
        'in wpp, micro batches num must equal to pipeline size '
    # num_warmup_microbatches = num_model_chunks * pipeline_parallel_size - \
    #     (pipeline_parallel_rank - pipeline_parallel_rank)
    num_forward_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank
    num_microbatches_remaining = pipeline_parallel_size - pipeline_parallel_rank
    num_forward_steady_microbatches = total_num_microbatches - num_forward_warmup_microbatches - num_microbatches_remaining

    # if torch.distributed.get_rank() == 0:
    # print(torch.distributed.get_rank(), 'num_warmup_microbatches', num_warmup_microbatches)
    rank = torch.distributed.get_rank()
    args = get_args()
    log_file_name = args.save+'/log%d.txt'%rank
    # with open(log_file_name, 'w') as log_file:
    #     log_file.write(f"pipeline_parallel_size: {pipeline_parallel_size}\n")
    #     log_file.write(f"pipeline_parallel_rank: {pipeline_parallel_rank}\n")
    #     log_file.write(f"""num microbatches\n
    #                    num_microbatches: {num_microbatches}\n
    #                    num_forward_warmup_microbatches: {num_forward_warmup_microbatches}\n
    #                    num_forward_steady_microbatches: {num_forward_steady_microbatches}\n
    #                    num_microbatches_remaining: {num_microbatches_remaining}\n""")
    
    recv_tensor_shapes = get_tensor_shapes(rank=rank-1,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel)
    send_tensor_shapes = get_tensor_shapes(rank=rank,
                                           model_type=model_type,
                                           tensor_shape=tensor_shape,
                                           decoder_seq_length=decoder_seq_length,
                                           sequence_parallel=sequence_parallel)
    # with open(log_file_name, 'a') as log_file:
    #     log_file.write(f"recv_tensor_shapes: {recv_tensor_shapes}\n")
    #     log_file.write(f"send_tensor_shapes: {send_tensor_shapes}\n")
    #     log_file.write(f"tensor_shape: {tensor_shape}\n")
    
    # Synchronize params for first two model chunks
    if param_sync_func is not None:
        param_sync_func(model[0].parameters())
        param_sync_func(model[1].parameters())

    def get_model_chunk_id(microbatch_id, forward):
        if forward:
            model_chunk_id = model_chunk_ids[microbatch_id]
            model_chunk_ids[microbatch_id] += 1
        else:
            model_chunk_id = (num_model_chunks - model_chunk_id - 1)
        return model_chunk_id

    def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == 0:
            return microbatch_id_in_group % pipeline_parallel_size == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == num_microbatch_groups - 1:
            return microbatch_id_in_group % pipeline_parallel_size == pipeline_parallel_size - 1
        else:
            return False

    def forward_step_helper(microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        # print('forward_step_helper, stage id:', torch.distributed.get_rank(),
        #       ', microbatch id:', microbatch_id)
        rank = torch.distributed.get_rank()
        args = get_args()
        log_file_name = args.save+'/log%d.txt'%rank
        # with open(log_file_name, 'a') as log_file:
        #     log_file.write(f"""forward_step_helper, time:{time.time()},
        #                    microbatch id: {microbatch_id}\n""")
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if param_sync_func is not None:
            param_sync_microbatch_id = microbatch_id + pipeline_parallel_rank
            if param_sync_microbatch_id < num_microbatches and is_first_microbatch_for_model_chunk(param_sync_microbatch_id):
                param_sync_chunk_id = get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
                if 1 < param_sync_chunk_id < num_model_chunks:
                    param_sync_func(model[param_sync_chunk_id].parameters())

        # forward step
        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == \
                    len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(forward_step_func,
                                     data_iterator[model_chunk_id],
                                     model[model_chunk_id],
                                     num_microbatches,
                                     input_tensor,
                                     forward_data_store,
                                     timers,
                                     collect_non_loss_data,
                                     dtype,
                                     enable_autocast)
        output_tensors[model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        # print('backward_step_helper, stage id:', torch.distributed.get_rank(),
        #       ', microbatch id:', microbatch_id)
        rank = torch.distributed.get_rank()
        args = get_args()
        log_file_name = args.save+'/log%d.txt'%rank
        with open(log_file_name, 'a') as log_file:
            log_file.write(f"""backward_step_helper, time:{time.time()}, 
                           microbatch id: {microbatch_id}\n""")
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch grad synchronization (default)
        if grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = \
            backward_step(grad_scaler,
                          input_tensor,
                          output_tensor,
                          output_tensor_grad,
                          model_type,
                          timers,
                          deallocate_pipeline_outputs)

        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if grad_sync_func is not None:
            grad_sync_microbatch_id = microbatch_id - pipeline_parallel_rank
            if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(grad_sync_microbatch_id):
                grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                enable_grad_sync()
                grad_sync_func(model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

        return input_tensor_grad
    
    memory = []
    memory.append((time.time(), torch.cuda.memory_allocated()))

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    # input_tensors[0].append(
    #     p2p_communication.recv_forward(tensor_shape,
    #                                    dtype=dtype,
    #                                    batch_p2p_comm=batch_p2p_comm,
    #                                    timers=timers))

    # TODO:注释说异步的计算时间会变长
    # TODO:但是同步的太难写了，准备先写个异步的

    
    fwd_wait_handles = None
    bwd_wait_handles = None
    
    memory.append((time.time(), torch.cuda.memory_allocated()))
    
    # wpp参考了chimera包含两条流水线，需要分别记录modelID和batchID
    modelID1 = 0
    batchID1 = 0
    modelID2 = 1
    batchID2 = 0
    
    for k in range(num_forward_warmup_microbatches):

        if modelID1  < modelID2:
            model_chunk_id = modelID1
            microbatchID = batchID1
            batchID1 += 1
        else:
            model_chunk_id = modelID2
            microbatchID = batchID2
            batchID2 += 1
            
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
        if model_chunk_id % 2 == 0:
            input_tensor = None
            if pipeline_parallel_rank != 0:
                input_tensor = p2p_communication.\
                    recv_forward(tensor_shape, dtype=dtype,
                                batch_p2p_comm=batch_p2p_comm, 
                                timers=timers)
        else:
            if pipeline_parallel_rank != (pipeline_parallel_size - 1):
                input_tensor = p2p_communication.\
                    recv_forward(tensor_shape, dtype=dtype,
                                batch_p2p_comm=batch_p2p_comm, 
                                timers=timers, from_prev=False)    
        if not forward_only:
            input_tensors[model_chunk_id].append(input_tensor)
                
        # with open(log_file_name, 'a') as log_file:
        #     log_file.write(f"""time: {time.time()}, forward_step: batchID{microbatchID}, modelID{model_chunk_id}\n""")\
                
        output_tensor = forward_step(forward_step_func, data_iterator[model_chunk_id], 
                                     model[model_chunk_id], num_microbatches,
                                     input_tensor, forward_data_store,
                                     timers, collect_non_loss_data, dtype, enable_autocast)
        if get_dependency:
            dependencies['forward'].append((model_chunk_id, microbatchID))
        memory.append((time.time(), torch.cuda.memory_allocated()))
        
        if model_chunk_id % 2 == 0:
            if pipeline_parallel_rank != (pipeline_parallel_size - 1):
                p2p_communication.send_forward(output_tensor,
                                            batch_p2p_comm=batch_p2p_comm,
                                            timers=timers)
            else:
                input_tensor = output_tensor
        else:
            if pipeline_parallel_rank != 0:
                p2p_communication.send_forward(output_tensor,
                                            batch_p2p_comm=batch_p2p_comm,
                                            timers=timers,
                                            to_next=False)
            else:
                input_tensor = output_tensor
        if not forward_only:
            output_tensors[model_chunk_id].append(output_tensor)
        if isinstance(output_tensor, list):
            deallocate_output_tensor(output_tensor[0], deallocate_pipeline_outputs)
        else:
            deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)
        
        if batchID1 == num_microbatches:
            batchID1 = 0
            modelID1 += 2
        if batchID2 == num_microbatches:
            batchID2 = 0
            modelID2 += 2
        
    parallel_state.set_virtual_pipeline_model_parallel_rank(modelID2)
    if pipeline_parallel_rank != (pipeline_parallel_size - 1):
        # with open(log_file_name, 'a') as log_file:
        #     log_file.write(f"   receiving from next stage\n")
        input_tensor = p2p_communication.recv_forward(tensor_shape,
                                    dtype=dtype,
                                    batch_p2p_comm=batch_p2p_comm,
                                    timers=timers,
                                    from_prev=False)
    # input_tensors[modelID2].append(input_tensor)
    # with open(log_file_name, 'a') as log_file:
    #     log_file.write(f"   received %s\n"%str(input_tensor==None))
    
    
    for k in range(0, num_forward_steady_microbatches, 2):
        microbatchID = batchID2
        model_chunk_id = modelID2
        batchID2 += 1
      
        
        # with open(log_file_name, 'a') as log_file:
        #     log_file.write(f"""time: {time.time()}, forward_step: batchID{microbatchID}, modelID{model_chunk_id}\n""")
        
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
        output_tensor = forward_step(forward_step_func, data_iterator[model_chunk_id],
                                        model[model_chunk_id], num_microbatches,
                                        input_tensor, forward_data_store,
                                        timers, collect_non_loss_data, dtype, enable_autocast)
        if not forward_only:
            input_tensors[model_chunk_id].append(input_tensor)
            output_tensors[model_chunk_id].append(output_tensor)
        if get_dependency:
            dependencies['forward'].append((model_chunk_id, microbatchID))
        memory.append((time.time(), torch.cuda.memory_allocated()))
        
        if pipeline_parallel_rank != 0:
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"   receiving from prev stage\n")
            parallel_state.set_virtual_pipeline_model_parallel_rank(modelID1)
            input_tensor = p2p_communication.recv_forward(tensor_shape,
                                        dtype=dtype,
                                        batch_p2p_comm=batch_p2p_comm,
                                        timers=timers) 
            
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"   received %s\n"%str(input_tensor==None))
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"   sending to prev stage %s\n"%(str(output_tensor==None)))
            parallel_state.set_virtual_pipeline_model_parallel_rank(modelID2)
            p2p_communication.send_forward(output_tensor,
                                            batch_p2p_comm=batch_p2p_comm,
                                            timers=timers,
                                            to_next=False)
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"   sent %s\n"%str(output_tensor))
        else:
            input_tensor = output_tensor
        # input_tensors[modelID1].append(input_tensor)
        # output_tensors[modelID2].append(output_tensor)
        if isinstance(output_tensor, list):
            deallocate_output_tensor(output_tensor[0], deallocate_pipeline_outputs)
        else:
            deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)
        
        # microbatchID = orderedID
        # orderedID = (orderedID + 1) % num_microbatches
        # model_chunk_id = get_model_chunk_id(microbatchID, forward=True)
        microbatchID = batchID1
        model_chunk_id = modelID1
        batchID1 += 1
             
                
        # with open(log_file_name, 'a') as log_file:
        #     log_file.write(f"""time: {time.time()}, forward_step: batchID{microbatchID}, modelID{model_chunk_id}\n""")
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
        output_tensor = forward_step(forward_step_func, data_iterator[model_chunk_id], 
                                     model[model_chunk_id], num_microbatches,
                                     input_tensor, forward_data_store,
                                     timers, collect_non_loss_data, dtype, enable_autocast)
        if get_dependency:
            dependencies['forward'].append((model_chunk_id, microbatchID))
        if not forward_only:
            input_tensors[model_chunk_id].append(input_tensor)    
            output_tensors[model_chunk_id].append(output_tensor)
        memory.append((time.time(), torch.cuda.memory_allocated()))
        
        
        if pipeline_parallel_rank != (pipeline_parallel_size - 1):
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"   sending to next stage %s\n"%(str(output_tensor==None)))
            p2p_communication.send_forward(output_tensor,
                                            batch_p2p_comm=batch_p2p_comm,
                                            timers=timers)
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"   receiving from next stage\n")
            parallel_state.set_virtual_pipeline_model_parallel_rank(modelID2)
            input_tensor = p2p_communication.recv_forward(tensor_shape,
                                        dtype=dtype,
                                        batch_p2p_comm=batch_p2p_comm,
                                        timers=timers,
                                        from_prev=False)             
        else:
            input_tensor = output_tensor
        # input_tensors[modelID2].append(input_tensor)
        # output_tensors[modelID1].append(output_tensor)
        if isinstance(output_tensor, list):
            deallocate_output_tensor(output_tensor[0], deallocate_pipeline_outputs)
        else:
            deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)
        # with open(log_file_name, 'a') as log_file:
        #     log_file.write(f"   received %s\n"%str(input_tensor==None))
            
        if batchID1 == num_microbatches:
            batchID1 = 0
            modelID1 += 2
        if batchID2 == num_microbatches:
            batchID2 = 0
            modelID2 += 2
        # with open(log_file_name, 'a') as log_file:
        #     log_file.write(f"modelID1: {modelID1}, batchID1: {batchID1}\n")
        #     log_file.write(f"modelID2: {modelID2}, batchID2: {batchID2}\n")        

    back_modelID1 = len(model) - 2
    back_batchID1 = 0
    back_modelID2 = len(model) - 1
    back_batchID2 = 0
    # with open(log_file_name, 'a') as log_file:
    #     log_file.write(f"---input tensors---\n")
    #     for l in range(len(input_tensors)):
    #         log_file.write(f"%d, len=%d\n"%(l, len(input_tensors[l])))
    #     log_file.write(f"------\n")
        
    for k in range(num_microbatches_remaining):
        microbatchID = batchID2
        model_chunk_id = modelID2
        batchID2 += 1
        
        # model_chunk_id = get_model_chunk_id(microbatchID, forward=True)
        
        # input_tensors[model_chunk_id].append(input_tensor)
        
        # with open(log_file_name, 'a') as log_file:
        #     log_file.write(f"""remain {k}\ntime: {time.time()}, forward_step: batchID{microbatchID}, modelID{model_chunk_id}\n""")
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
        output_tensor = forward_step(forward_step_func, data_iterator[model_chunk_id],
                                        model[model_chunk_id], num_microbatches,
                                        input_tensor, forward_data_store,
                                        timers, collect_non_loss_data, dtype, enable_autocast)
        if not forward_only:
            input_tensors[model_chunk_id].append(input_tensor)
            output_tensors[model_chunk_id].append(output_tensor)
        if get_dependency:
            dependencies['forward'].append((model_chunk_id, microbatchID))
        memory.append((time.time(), torch.cuda.memory_allocated()))
        
        # with open(log_file_name, 'a') as log_file:
        #     log_file.write(f"   sending & receiving with pre stage\n ")
        if pipeline_parallel_rank == 0:
            output_tensor_grad = None
        else:
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"   sending %s\n"%str(output_tensor==None))
            p2p_communication.send_forward(output_tensor, 
                                            batch_p2p_comm, 
                                            timers=timers,
                                            to_next=False)
            if not forward_only:
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   receiving\n")
                parallel_state.set_virtual_pipeline_model_parallel_rank(back_modelID2)
                output_tensor_grad = p2p_communication.recv_backward(tensor_shape,
                                                                    dtype=dtype,
                                                                    batch_p2p_comm=batch_p2p_comm,
                                                                    timers=timers,
                                                                    from_next=False)
        # if not forward_only:
        #     with open(log_file_name, 'a') as log_file:
        #         log_file.write(f"   received %s\n"%(output_tensor_grad==None))
            
        # output_tensors[modelID2].append(output_tensor)
        deallocate_output_tensor(output_tensor, deallocate_pipeline_outputs)
        # output_tensor_grads[back_modelID2].append(output_tensor_grad)
        
        if not forward_only:
            microbatchID = back_batchID2
            model_chunk_id = back_modelID2
            back_batchID2 += 1
            
            input_tensor = input_tensors[back_modelID2].pop(0)
            output_tensor = output_tensors[back_modelID2].pop(0)
            
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"""time: {time.time()}, backward_step: batchID{microbatchID}, modelID{model_chunk_id}\n""")
            #     log_file.write(f"   output tensor grad %s\n"%str(output_tensor_grad))
                
            parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
            input_tensor_grad = backward_step(grad_scaler,
                            input_tensor,
                            output_tensor,
                            output_tensor_grad,
                            model_type,
                            timers,
                            deallocate_pipeline_outputs)
            if get_dependency:
                dependencies['backward'].append((model_chunk_id, microbatchID))
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"   input tensor grad %s\n"%str(input_tensor_grad))
        
            if k == (num_microbatches_remaining - 1):
                if pipeline_parallel_rank != (pipeline_parallel_size - 1):
                    p2p_communication.send_backward(input_tensor_grad, 
                                                    batch_p2p_comm, 
                                                    timers=timers,
                                                    to_prev=False)
                    parallel_state.set_virtual_pipeline_model_parallel_rank(back_modelID1)
                    output_tensor_grad = p2p_communication.\
                        recv_backward(tensor_shape, dtype=dtype, timers=timers)
                else:
                    output_tensor_grad = input_tensor_grad
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   received output tensor grad %s\n"%str(output_tensor_grad))
            else:
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   sending & receiving with next stage\n")
                parallel_state.set_virtual_pipeline_model_parallel_rank(modelID2)
                input_tensor = p2p_communication.recv_forward(tensor_shape,
                                                                dtype=dtype,
                                                                batch_p2p_comm=batch_p2p_comm,
                                                                timers=timers,
                                                                from_prev=False)
                parallel_state.set_virtual_pipeline_model_parallel_rank(back_modelID2)
                p2p_communication.send_backward(input_tensor_grad,
                                                batch_p2p_comm,
                                                timers=timers,
                                                to_prev=False)
                # input_tensors[modelID2].append(input_tensor)
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   received %s\n"%(input_tensor==None))
                #     if isinstance(input_tensor, list):
                #         log_file.write(f"   input tensor %s %d\n"%(str(input_tensor[0]==None), len(input_tensor)))
        else:
            if k != (num_microbatches_remaining - 1):
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   receiving %s\n")
                parallel_state.set_virtual_pipeline_model_parallel_rank(modelID2)
                input_tensor = p2p_communication.recv_forward(tensor_shape,
                                                                dtype=dtype,
                                                                batch_p2p_comm=batch_p2p_comm,
                                                                timers=timers,
                                                                from_prev=False)
            
    if not forward_only:        
        if back_batchID2 == num_microbatches:
            back_batchID2 = 0
            back_modelID2 -= 2 

        for k in range(0, num_forward_steady_microbatches, 2):
            microbatchID = back_batchID1
            model_chunk_id = back_modelID1
            back_batchID1 += 1
            # model_chunk_id = get_model_chunk_id(k+num_forward_warmup_microbatches, forward=True)
                            
            input_tensor = input_tensors[model_chunk_id].pop(0)
            output_tensor = output_tensors[model_chunk_id].pop(0)
            
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"""time: {time.time()}, backward_step: batchID{microbatchID}, modelID{model_chunk_id}\n""")

            parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
            if pipeline_parallel_rank != (pipeline_parallel_size - 1):
                input_tensor_grad = backward_step(grad_scaler,
                                                input_tensor,
                                                output_tensor,
                                                output_tensor_grad,
                                                model_type,
                                                timers,
                                                deallocate_pipeline_outputs)
            if get_dependency:
                dependencies['backward'].append((model_chunk_id, microbatchID))                                    
            memory.append((time.time(), torch.cuda.memory_allocated()))
            
            if pipeline_parallel_rank != 0:
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   receiving %s\n")
                parallel_state.set_virtual_pipeline_model_parallel_rank(back_modelID2)
                output_tensor_grad = \
                    p2p_communication.recv_backward(tensor_shape,
                                                    dtype=dtype,
                                                    batch_p2p_comm=batch_p2p_comm,
                                                    timers=timers,
                                                    from_next=False)
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   received %s\n"%(output_tensor_grad==None))
                #     log_file.write(f"   sending %s\n"%(input_tensor_grad==None))
                parallel_state.set_virtual_pipeline_model_parallel_rank(back_modelID1)
                p2p_communication.send_backward(input_tensor_grad, batch_p2p_comm, timers=timers)
            else:
                output_tensor_grad = input_tensor_grad
            
            microbatchID = back_batchID2
            model_chunk_id = back_modelID2
            back_batchID2 += 1
            
            input_tensor = input_tensors[model_chunk_id].pop(0)
            output_tensor = output_tensors[model_chunk_id].pop(0)
            
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"""time: {time.time()}, backward_step: batchID{microbatchID}, modelID{model_chunk_id}\n""")
            parallel_state.set_virtual_pipeline_model_parallel_rank(back_modelID2)
            if pipeline_parallel_rank != 0:
                input_tensor_grad = backward_step(grad_scaler,
                                                input_tensor,
                                                output_tensor,
                                                output_tensor_grad,
                                                model_type,
                                                timers,
                                                deallocate_pipeline_outputs)            
            if get_dependency:
                dependencies['backward'].append((model_chunk_id, microbatchID))
            if pipeline_parallel_rank != (pipeline_parallel_size - 1):
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   sending %s\n"%(input_tensor_grad==None))
                p2p_communication.send_backward(input_tensor_grad, batch_p2p_comm, 
                                                timers=timers, to_prev=False)
                # with open(log_file_name, 'a') as log_file:
                #     log_file.write(f"   receiving %s\n")
                parallel_state.set_virtual_pipeline_model_parallel_rank(back_modelID1)
                output_tensor_grad = p2p_communication.\
                    recv_backward(tensor_shape, dtype=dtype, timers=timers)
            else:
                output_tensor_grad = input_tensor_grad
            # with open(log_file_name, 'a') as log_file:
            #         log_file.write(f"   received %s\n"%(output_tensor_grad==None))
            if back_batchID1 == num_microbatches:
                back_batchID1 = 0
                back_modelID1 -= 2
            if back_batchID2 == num_microbatches:
                back_batchID2 = 0
                back_modelID2 -= 2
                
        for k in range(num_forward_warmup_microbatches):
            microbatchID = back_batchID1
            model_chunk_id = back_modelID1
            back_batchID1 += 1
            
            input_tensor = input_tensors[model_chunk_id].pop(0)
            output_tensor = output_tensors[model_chunk_id].pop(0)
            
            # with open(log_file_name, 'a') as log_file:
            #     log_file.write(f"""time: {time.time()}, backward_step: batchID{microbatchID}, modelID{model_chunk_id}\n""")
            parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)
            if pipeline_parallel_rank != (pipeline_parallel_size - 1):
                input_tensor_grad = backward_step(grad_scaler,
                                                input_tensor,
                                                output_tensor,
                                                output_tensor_grad,
                                                model_type,
                                                timers,
                                                deallocate_pipeline_outputs) 
            if get_dependency:
                dependencies['backward'].append((model_chunk_id, microbatchID))
            p2p_communication.send_backward(input_tensor_grad, batch_p2p_comm, timers=timers)
            if k != (num_forward_warmup_microbatches-1):
                output_tensor_grad = p2p_communication.\
                    recv_backward(tensor_shape, dtype=dtype, timers=timers)
        
    enable_grad_sync()
    if grad_sync_func is not None:
        params = []
        for model_chunk_id in range(num_model_chunks):
            if model_chunk_id not in synchronized_model_chunks:
                params.extend(model[model_chunk_id].parameters())
                synchronized_model_chunks.add(model_chunk_id)
        if params:
            grad_sync_func(params)
            
    memory.append((time.time(), torch.cuda.memory_allocated()))
    
    args = get_args()
    record_path = args.save+'/memorylog%d.json'%torch.distributed.get_rank()
    with open(record_path, 'r') as file:
        data = json.load(file)
    data['memory'].extend(memory)
    with open(record_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    for i in range(len(input_tensors)):
        assert len(input_tensors[i]) == 0,\
            f"input_tensors[{i}] is not empty"
    for i in range(len(output_tensors)):
        assert len(output_tensors[i]) == 0,\
            f"output_tensors[{i}] is not empty"

    return forward_data_store, dependencies

