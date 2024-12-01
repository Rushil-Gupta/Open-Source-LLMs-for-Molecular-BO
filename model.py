from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.chat_models import ChatHuggingFace
import torch


class Model:
    """Model provides an abstraction around chat and completion models suitable for this
    project, so that we can use chat and completion models with the same code."""

    name: str

    def __init__(self, model: BaseLanguageModel, chat: bool = False):
        self.model = model
        self.chat = chat

    @classmethod
    def from_full_name(cls, full_name: str, temp: float, **kwargs):
        """Create a new model from a full name, which includes the service and model name.

        The full_name parameter combines the name of an LLM service (e.g. "hf") with the name of
        a model (e.g. "llama3"), separated by a hyphen (e.g. "hf-llama3").

        """
        service, model = full_name.lower().split("-", 1)

        if service == "hf":
            model_id = kwargs["model_path"] + "/" + model
            rank = kwargs["rank"]
            llm = HuggingFacePipeline.from_model_id(
                model_id=model_id,
                task="text-generation",
                model_kwargs={"torch_dtype": torch.bfloat16, "device_map": rank},
                device=None,
                batch_size=256,
                pipeline_kwargs=dict(
                    max_new_tokens=12000,
                    temperature=temp,
                ),
            )
            model = ChatHuggingFace(llm=llm)
            if full_name == "hf-llama3.1-8b-instruct":
                model.llm.pipeline.tokenizer.pad_token_id = (
                    model.llm.pipeline.tokenizer.eos_token_id
                )
        else:
            raise NotImplementedError(f"Model {full_name} not implemented.")

        out = cls(model, True)
        out.name = full_name
        return out

    def build_prompt(
        self,
        prompt: str,
        context: str | None = None,
    ) -> str | list[BaseMessage]:
        """Builds the full input to the LLM as a list of messages for a
        chat model.

        If context is provided, it is added before the prompt.
        For a text completion model, the full prompt looks like this:

            {context}
            {prompt}

        For a chat model, the messages look like this:

            System: {context}
            Human: {prompt}

        The type of the output depends on whether this is a chat or completion model.
        """
        if self.chat:
            return self.build_chat_prompt(prompt, context)

        return self.build_completion_prompt(prompt, context)

    @staticmethod
    def build_chat_prompt(
        prompt: str,
        context: str | None = None,
    ) -> list[BaseMessage]:
        out = []

        if context:
            out.append(SystemMessage(content=context))

        out.append(HumanMessage(content=prompt))

        return out

    def build_completion_prompt(
        self,
        prompt: str,
        context: str | None = None,
    ) -> str:
        out = ""

        if context:
            out = context + "\n\n"

        out += prompt + "\n"

        return out

    def generate(self, final_prompt: str | list[BaseMessage], **kwargs) -> str:
        """Call the model with the prepared prompt and return the response text."""

        with torch.no_grad():
            message = self.model.invoke(final_prompt, **kwargs)

        if self.chat:
            return message.content

        return message
