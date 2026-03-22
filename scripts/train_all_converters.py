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
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    print("=" * 60)
    print("Training entity converter")
    print("=" * 60)
    import apollo.entity_minilm_converter as entity_minilm_converter

    entity_minilm_converter.main()

    print()
    print("=" * 60)
    print("Training context converter")
    print("=" * 60)
    import apollo.context_minilm_converter as context_minilm_converter

    context_minilm_converter.main()

    print()
    print("=" * 60)
    print("Training personality converter")
    print("=" * 60)
    import apollo.personality_minilm_converter as personality_minilm_converter

    personality_minilm_converter.main()

    print()
    print("All converters trained.")


if __name__ == "__main__":
    main()
    sys.exit(0)
