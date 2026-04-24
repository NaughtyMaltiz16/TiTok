def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def get_output(output, args):
    def strip_quotes(text):
        return text.strip().strip('"').strip("'")

    model_name = args.model_name.lower()
    lines = output.strip().splitlines()

    if "mistral" in model_name:
        result = output.strip()
    else:
        # Llama3 and others
        candidate_lines = [line.strip() for line in lines if line.strip().startswith('"') or line.strip().endswith('"')]
        result = candidate_lines[-1] if candidate_lines else lines[-1].strip()

    return strip_quotes(result)