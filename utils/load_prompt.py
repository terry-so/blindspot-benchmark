
def load_prompt_template(path: str) -> str:
    with open(path, "r") as f:
        return f.read()