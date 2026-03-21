"""
train_all_converters.py

Trains MiniLM <-> embedding converters for all three categories:
  entity, context, personality.

Requires:
  - entity_embeddings/entity_embeddings.pt + vocab.json
  - context_embeddings/context_embeddings.pt + vocab.json
  - personality_embeddings/personality_embeddings.pt + vocab.json

Run the respective train_*_embeddings.py scripts first if needed.
"""

import sys


def main() -> None:
    print("=" * 60)
    print("Training entity converter")
    print("=" * 60)
    import entity_minilm_converter

    entity_minilm_converter.main()

    print()
    print("=" * 60)
    print("Training context converter")
    print("=" * 60)
    import context_minilm_converter

    context_minilm_converter.main()

    print()
    print("=" * 60)
    print("Training personality converter")
    print("=" * 60)
    import personality_minilm_converter

    personality_minilm_converter.main()

    print()
    print("All converters trained.")


if __name__ == "__main__":
    main()
    sys.exit(0)
