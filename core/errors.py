"""Root of the core error hierarchy.

It exists because of a real defect found in review: the recall hook caught
`EmbeddingError` and `QdrantError`, but the MOST COMMON failure — an unreachable
endpoint — arrived as `HttpError`, which was neither. The exact result: a traceback
for the USER and silence for the MODEL, i.e. the inverse of the contract the hook
exists to fulfil.

The lesson is not "a type was missing from the list": it is that a list of types to
catch is fragile by construction — it has to be updated in every consumer each time
a new error appears, and forgetting does not produce a compile error, it produces
silence in production. With a root, `except CoreError` is correct BY CONSTRUCTION,
and a new error is caught the day it is born.
"""


class CoreError(Exception):
    """Any expected failure of the core.

    Consumers should catch THIS. Catching specific subclasses is for when the message
    to the user changes with the type — never for deciding WHETHER the failure is
    handled.
    """


#: The failures that are about the ENVIRONMENT, not about the input — an endpoint that is
#: down, an archive that cannot be reached. Imported lazily by the modules that raise them,
#: so this root stays dependency-free.
#:
#: WHY THE DISTINCTION HAS TO EXIST SOMEWHERE. A caller that handles one path at a time —
#: `repos.add_files` — cannot treat these two families alike: "this file has nothing
#: indexable" is a permanent property of the file, while "the embedding endpoint timed out"
#: is a property of the minute. Recording the second as if it were the first is how a
#: network blip becomes a permanent exclusion, and the indexing quarantine made that
#: consequence durable rather than merely annoying.
def infrastructure_errors() -> tuple:
    """The error types that mean "the environment failed", never "this input is bad"."""
    from .embedding import EmbeddingError
    from .http import HttpError
    from .qdrant import QdrantError
    from .reranking import RerankError

    return (EmbeddingError, HttpError, QdrantError, RerankError)
