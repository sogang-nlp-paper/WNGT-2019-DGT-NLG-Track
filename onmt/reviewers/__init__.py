from onmt.reviewers.reviewer import InputReviewer, OutputReviewer


str2review = {"input": InputReviewer, "output": OutputReviewer}

__all__ = ["InputReviewer", "OutputReviewer"]