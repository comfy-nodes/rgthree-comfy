import asyncio

from .constants import get_category, get_name
from .power_prompt_utils import get_lora_by_filename
from .utils import FlexibleOptionalInputType, any_type
from .server.utils_info import get_model_info
from .log import log_node_warn

NODE_NAME = get_name("Power Lora Stacker")


class RgthreePowerLoraStacker:
  """The Power Lora Stacker is similar to the Power Lora Loader, but has the LORA_STACK output instead of MODEL and CLIP."""

  NAME = NODE_NAME
  CATEGORY = get_category()

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {},
      # Since we will pass any number of loras in from the UI, this needs to always allow an
      "optional": FlexibleOptionalInputType(type=any_type, data={
        # "lora_stack": ("LORA_STACK", {"default": None}),
      }),
      "hidden": {},
    }

  RETURN_TYPES = ("LORA_STACK",)
  RETURN_NAMES = ("LORA_STACK",)
  FUNCTION = "load_loras"

  def load_loras(self, **kwargs):
    """Loops over the provided loras in kwargs and applies valid ones."""
    lora_stack = kwargs.get("lora_stack", list())
    for key, value in kwargs.items():
      key = key.upper()
      if key.startswith("LORA_") and "on" in value and "lora" in value and "strength" in value:
        strength_model = value["strength"]
        # If we just passed one strtength value, then use it for both, if we passed a strengthTwo
        # as well, then our `strength` will be for the model, and `strengthTwo` for clip.
        strength_clip = (
          value["strengthTwo"]
          if "strengthTwo" in value and value["strengthTwo"] is not None
          else strength_model
        )
        if value["on"] and (strength_model != 0 or strength_clip != 0):
          lora = get_lora_by_filename(value["lora"], log_node=self.NAME)
          if lora is not None:
            if lora_stack is not None:
              lora_stack.extend([(lora, strength_model, strength_clip)])

    return [lora_stack]

  @classmethod
  def get_enabled_loras_from_prompt_node(cls, prompt_node: dict) -> list[dict[str, str | float]]:
    return [{
      'name': lora['lora'],
      'strength': lora['strength']
    } | ({
      'strength_clip': lora['strengthTwo']
    } if 'strengthTwo' in lora else {})
            for name, lora in prompt_node['inputs'].items()
            if name.startswith('lora_') and lora['on']]

  @classmethod
  def get_enabled_triggers_from_prompt_node(cls, prompt_node: dict, max_each: int = 1):
    loras = [l['name'] for l in cls.get_enabled_loras_from_prompt_node(prompt_node)]
    trained_words = []
    for lora in loras:
      info = asyncio.run(get_model_info(lora, 'loras'))
      if not info or not info.keys():
        log_node_warn(NODE_NAME, f'No info found for lora {lora} when grabbing triggers.')
        continue
      if 'trainedWords' not in info or not info['trainedWords']:
        log_node_warn(NODE_NAME, f'No trained words for lora {lora} when grabbing triggers.')
        continue
      trained_words += [w for wi in info['trainedWords'][:max_each] if (wi and (w := wi['word']))]
    return trained_words
