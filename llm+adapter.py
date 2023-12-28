{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyP8zUi5zuxb9laE1XpXzNU1",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/manas95826/fInE_tUnE-with_themanas/blob/main/llm%2Badapter.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QmSCDFVm21Rt",
        "outputId": "3f611958-a34f-411c-e282-96edbae2da47"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m16.6/16.6 MB\u001b[0m \u001b[31m36.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.3/1.3 MB\u001b[0m \u001b[31m31.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m168.3/168.3 kB\u001b[0m \u001b[31m13.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m93.1/93.1 kB\u001b[0m \u001b[31m9.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m305.1/305.1 kB\u001b[0m \u001b[31m25.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m75.9/75.9 kB\u001b[0m \u001b[31m7.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m138.7/138.7 kB\u001b[0m \u001b[31m3.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m381.9/381.9 kB\u001b[0m \u001b[31m12.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m45.7/45.7 kB\u001b[0m \u001b[31m2.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m60.3/60.3 kB\u001b[0m \u001b[31m5.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m129.9/129.9 kB\u001b[0m \u001b[31m6.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m265.7/265.7 kB\u001b[0m \u001b[31m12.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2.1/2.1 MB\u001b[0m \u001b[31m9.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m58.3/58.3 kB\u001b[0m \u001b[31m2.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m67.0/67.0 kB\u001b[0m \u001b[31m3.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m76.9/76.9 kB\u001b[0m \u001b[31m5.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Building wheel for ffmpy (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
            "lida 0.0.10 requires kaleido, which is not installed.\n",
            "llmx 0.0.15a0 requires cohere, which is not installed.\n",
            "llmx 0.0.15a0 requires openai, which is not installed.\n",
            "llmx 0.0.15a0 requires tiktoken, which is not installed.\n",
            "tensorflow-probability 0.22.0 requires typing-extensions<4.6.0, but you have typing-extensions 4.9.0 which is incompatible.\u001b[0m\u001b[31m\n",
            "\u001b[0m"
          ]
        }
      ],
      "source": [
        "!pip -q install transformers gradio sentencepiece peft"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import gradio as gr\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "from peft import PeftModel\n",
        "import torch\n",
        "\n",
        "\n",
        "def merge(base_model, trained_adapter, token):\n",
        "    base = AutoModelForCausalLM.from_pretrained(\n",
        "        base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True, token=token\n",
        "    )\n",
        "    model = PeftModel.from_pretrained(base, trained_adapter, token=token)\n",
        "    try:\n",
        "        tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)\n",
        "    except RecursionError:\n",
        "        tokenizer = AutoTokenizer.from_pretrained(\n",
        "            base_model, unk_token=\"<unk>\", token=token\n",
        "        )\n",
        "\n",
        "    model = model.merge_and_unload()\n",
        "\n",
        "    print(\"Saving target model\")\n",
        "    model.push_to_hub(trained_adapter, token=token)\n",
        "    tokenizer.push_to_hub(trained_adapter, token=token)\n",
        "    return gr.Markdown.update(\n",
        "        value=\"Model successfully merged and pushed! Please shutdown/pause this space\"\n",
        "    )\n",
        "\n",
        "\n",
        "with gr.Blocks() as demo:\n",
        "    gr.Markdown(\"## AutoTrain Merge Adapter\")\n",
        "    gr.Markdown(\"Please duplicate this space and attach a GPU in order to use it.\")\n",
        "    token = gr.Textbox(\n",
        "        label=\"Hugging Face Write Token\",\n",
        "        value=\"\",\n",
        "        lines=1,\n",
        "        max_lines=1,\n",
        "        interactive=True,\n",
        "        type=\"password\",\n",
        "    )\n",
        "    base_model = gr.Textbox(\n",
        "        label=\"Base Model (e.g. meta-llama/Llama-2-7b-chat-hf)\",\n",
        "        value=\"\",\n",
        "        lines=1,\n",
        "        max_lines=1,\n",
        "        interactive=True,\n",
        "    )\n",
        "    trained_adapter = gr.Textbox(\n",
        "        label=\"Trained Adapter Model (e.g. username/autotrain-my-llama)\",\n",
        "        value=\"\",\n",
        "        lines=1,\n",
        "        max_lines=1,\n",
        "        interactive=True,\n",
        "    )\n",
        "    submit = gr.Button(value=\"Merge & Push\")\n",
        "    op = gr.Markdown()\n",
        "    submit.click(merge, inputs=[base_model, trained_adapter, token], outputs=[op])\n",
        "\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    demo.launch()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 649
        },
        "id": "EpJt4QkC2-WN",
        "outputId": "3cdd7737-6a28-4ad3-c580-33322271b326"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Setting queue=True in a Colab notebook requires sharing enabled. Setting `share=True` (you can turn this off by setting `share=False` in `launch()` explicitly).\n",
            "\n",
            "Colab notebook detected. To show errors in colab notebook, set debug=True in launch()\n",
            "Running on public URL: https://b98f5a9bfe6544a112.gradio.live\n",
            "\n",
            "This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<div><iframe src=\"https://b98f5a9bfe6544a112.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "41OB5fwX3CUr"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}