---
title: "Add Cohere model config in optimum"
date: 2024-12-02
---
## Add Cohere model config in optimum 
**Step1** implement a custom ONNX configuration, if the model's architecture is similar to an architecture that is already spported, trying to simply inheriting from this class might work. For Aya-23-8B , I chose to inherit from GPTBigCodeOnnxConfig.

In file `optimum/optimum/exporters/onnx/model_configs.py`,the class `GPTBigCodeOnnxConfig`already exsits, and it inherits other classes, there are many other classes as well in this file

```python 
  class GPTBigCodeOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (
        GPTBigCodeDummyPastKeyValuesGenerator,
    ) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DEFAULT_ONNX_OPSET = 14  # GPT BigCode now uses F.scaled_dot_product_attention by default for torch>=2.1.1.
    DUMMY_PKV_GENERATOR_CLASS = GPTBigCodeDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("gpt_bigcode")

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_sequence_length + sequence_length"
            name = "present"

        for i in range(self._normalized_config.num_layers):
            # No dim for `n_head` when using multi-query attention
            inputs_or_outputs[f"{name}.{i}.key_value"] = {
                0: "batch_size",
                1: decoder_sequence_name,
            }

    def flatten_past_key_values(self, flattened_output, name, idx, t):
        flattened_output[f"{name}.{idx}.key_value"] = t
```

Then right after this class, I add another class for Cohere model, see below

```python
class CohereOnnxConfig(GPTBigCodeOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    ATOL_FOR_VALIDATION = 1e-4

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {
            "input_ids": dynamic_axis,
            "attention_mask": dynamic_axis,
            "position_ids": dynamic_axis,
        }
```

Two class attributes are specified here:
`NORMALIZED_CONFIG_CLASS`: this must be a NormalizedConfig, it basically allows the input generator to access the model config attributes in a generic way.
`ATOL_FOR_VALIDATION`: it is used when validating the exported model against the original one, this is the absolute acceptable tolerance for the output values difference.

It says in the document that ***Every configuration object must implement the inputs property and return a mapping, where each key corresponds to an input name, and each value indicates the axes in that input that are dynamic*** Thats why I add the inputs function in the class.

**Step2** 
Every type of model has its base model configuration, check [transformer/Cohere](https://huggingface.co/docs/transformers/main/en/model_doc/cohere)
After we implemented the CohereOnnxConfig in Step1, we can instantiate it by providing the base model's configuration as follows:
Here, I create aya_export.py script for exporting aya model
```python
from optimum.exporters.onnx.model_configs import CohereOnnxConfig
from transformers import CohereConfig, AutoModelForCausalLM

model_id = "CohereForAI/aya-23-8B"
config = CohereConfig.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
custom_cohere_onnx_config = CohereOnnxConfig(config=config, task="text-generation", use_past=True)
```
And then I printed out stuff below to see if it makes sense
```python
print(config) # This config is useful for later when we use the model to run evaluation, we need a config.json, this can be put there
print(custom_cohere_onnx_config.DEFAULT_ONNX_OPSET)
print(custom_cohere_onnx_config.outputs)
print(custom_cohere_onnx_config.inputs)
```
**Step3** Registering the ONNX configuration in the TasksManager. The TasksManager is the main entry-point to load a model given a name and a task, and to get the proper configuration for a given (architecture, backend) couple. When adding support for the export to ONNX, registering the configuration to the TasksManager will make the export available in the command line tool.

This is the file `optimum/exporters/tasks.py`, we add an entry in the `_SUPPORTED_MODEL_TYPE` attribute, see below:
```python
"cohere": supported_tasks_mapping(
            "text-generation",
            "text-generation-with-past",
            onnx="CohereOnnxConfig",
        ),
```
Now we have the model and the tasks it supports registered in tasks manager.In `optimum/utils/normalized_config.py` , in class `NormalizedConfigManager` we add an entry as well: ` "cohere": GPTBigCodeNormalizedTextConfig`

**Step4** Export onnx model
``` python
from transformers import AutoModelForCausalLM
from pathlib import Path
from optimum.exporters import TasksManager
from optimum.exporters.onnx import export
from pathlib import Path

model_id = "CohereForAI/aya-23-8B"
model = AutoModelForCausalLM.from_pretrained(model_id)

#Export
onnx_path = Path("models/CohereForAI/aya-23-8B/model.onnx")
onnx_config_constructor = TasksManager.get_exporter_config_constructor("onnx", model, task="text-generation-with-past")
onnx_config = onnx_config_constructor(model.config)
onnx_inputs, onnx_outputs = export(model, onnx_config, onnx_path, onnx_config.DEFAULT_ONNX_OPSET)
```
`TasksManager.get_exporter_config_constructor` gets the config constructor for a model type and task combination.

Then by excuting the file aya_export, I can get model.onnx and model.onnx_data. Remember to comment the code below (you can comment after it complains the error, its hard to find this file)
```python
if GLOBALS.onnx_shape_inference:
        _C._jit_pass_onnx_graph_shape_type_inference(

```
in `File "/opt/conda/lib/python3.10/site-packages/torch/onnx/utils.py", line 663, in _optimize_graph`, this code snippet shows twice in this file, we need to comment both of them, to sove this issure 

`RuntimeError: The serialized model is larger than the 2GiB limit imposed by the protobuf library. Therefore the output file must be a file path, so that the ONNX external data can be written to the same directory. Please specify the output file name.`

All the information is learned from the original document for exporting a model [optimum](https://huggingface.co/docs/optimum/en/exporters/onnx/usage_guides/contribute) 
The code is in branch fix/add-cohere-support-v2 or fix/add-cohere-onnx-support, they are pretty much the same.


