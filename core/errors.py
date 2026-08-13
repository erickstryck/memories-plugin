"""Raiz da hierarquia de erros do núcleo.

Existe por causa de um defeito real, encontrado em revisão: o hook de recall
capturava `EmbeddingError` e `QdrantError`, mas a falha MAIS COMUM — endpoint
inalcançável — chegava como `HttpError`, que não era nenhum dos dois. Resultado
exato: traceback para o USUÁRIO e silêncio para o MODELO, ou seja o inverso do
contrato que o hook existe para cumprir.

A lição não é "faltou um tipo na lista": é que uma lista de tipos a capturar é
frágil por construção — ela precisa ser atualizada em todo consumidor cada vez que
um erro novo aparece, e o esquecimento não dá erro de compilação, dá silêncio em
produção. Com uma raiz, `except CoreError` fica correto POR CONSTRUÇÃO, e erro novo
já nasce capturado.
"""


class CoreError(Exception):
    """Qualquer falha esperada do núcleo.

    Consumidores devem capturar ISTO. Capturar subclasses específicas é para quando
    a mensagem ao usuário muda de acordo com o tipo — nunca para decidir SE a falha
    é tratada.
    """
