def chunk_text(text: str, max_size: int = 3500) -> list[str]:
    if len(text) <= max_size:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_size:
            chunks.append(remaining)
            break

        split_at = remaining.rfind("\n", 0, max_size)
        if split_at == -1 or split_at < max_size // 2:
            split_at = max_size

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")

    return chunks
