from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        state_of_the_union = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([state_of_the_union])

    return [text for text in texts]
