import argparse
from operator import xor
from functools import reduce

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sentences_generation",
        action="store_true",
        help="Generate sentences from the datasets",
    )
    parser.add_argument(
        "--clustering_embeddings",
        action="store_true",
        help="Cluster the embeddings",
    )
    parser.add_argument(
        "--extract_sentences_with_target",
        action="store_true",
        help="Extract sentences with target",
    )
    parser.add_argument(
        "--salience_extraction",
        action="store_true",
        help="Extract salience",
    )
    parser.add_argument(
        "--training_classifier",
        action="store_true",
        help="Train classifier",
    )

    # add usage examples
    parser.usage = """
    python -m marc_thesis --sentences_generation
    python -m marc_thesis --clustering_embeddings
    python -m marc_thesis --extract_sentences_with_target
    """

    args = parser.parse_args()

    # Check if at least one argument is passed
    if not reduce(xor, vars(args).values()):
        parser.print_help()
        exit(1)

    if args.sentences_generation:
        import marc_thesis.sentence_generation.main as sentence_generation

        sentence_generation.main()

    elif args.extract_sentences_with_target:
        import marc_thesis.extract_sentences_with_target.main as extract_sentences_with_target

        extract_sentences_with_target.main()

    elif args.clustering_embeddings:
        import marc_thesis.clustering_embeddings.main as clustering_embeddings

        clustering_embeddings.main()

    elif args.salience_extraction:
        import marc_thesis.salience_extraction.main as salience_extraction

        salience_extraction.main()

    elif args.training_classifier:
        import marc_thesis.training_classifier.main as training_classifier

        training_classifier.main()
