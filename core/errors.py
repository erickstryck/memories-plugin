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
