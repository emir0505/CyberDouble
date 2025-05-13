from unsloth import FastLanguageModel
from unsloth import     
from trl import SFTTrainer, SFTConfig, apply_chat_template, setup_chat_format
from transformers import AutoTokenizer
from datasets import load_dataset

if __name__ == "__main__":
    # model_id = "unsloth/phi-4"
    # model_id = "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit"
    model_id = "unsloth/Phi-3.5-mini-instruct-bnb-4bit"
    max_seq_length = 2048

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id, max_seq_length=max_seq_length, load_in_4bit=True
    )
    # tokenizer = AutoTokenizer.from_pretrained(model_id)

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        tokenizer.chat_template = None  # Reset the chat template

    model, tokenizer = setup_chat_format(model, tokenizer)

    lora_alpha = 16  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    dataset = load_dataset("json", data_files="./data.jsonl", split="train")
    print(dataset)

    def convert(example):
        res = apply_chat_template(example=example, tokenizer=tokenizer)
        return res

    dataset = dataset.map(convert, remove_columns="messages")
    print("dataset[0]:", dataset[0])

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=160,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            output_dir="outputs",
            optim="adamw_8bit",
            seed=3407,
            dataset_num_proc=1,
            dataloader_num_workers=0,
        ),
    )
    trainer.train()

    save_path = "adapter2"
    model.save_pretrained_merged(save_path, tokenizer, save_method="merged_16bit")
    tokenizer.save_pretrained(save_path)
    # model.save_pretrained_gguf(save_path, tokenizer, quantization_method="f16")
    print(f"LoRA adapters saved to {save_path}")

print("Done!")
