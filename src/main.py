from src.graph_impl import Graph
from vertex import Vertex


def main():
    print("=== INICIANDO TESTES DO GRAFO ===\n")

    # 1. Instanciação do Grafo
    g = Graph()
    print(f"Grafo criado. É vazio? {g.is_empty_graph()}")

    # 2. Criação e Adição de Vértices
    # Nota: O construtor do Vertex pede (nome, peso, lista_de_arestas)
    # Inicializamos com lista vazia []
    v0 = Vertex("A", 1.0, [])
    v1 = Vertex("B", 2.0, [])
    v2 = Vertex("C", 3.0, [])
    v3 = Vertex("D", 4.0, [])

    g.set_vertex(v0)  # Índice 0
    g.set_vertex(v1)  # Índice 1
    g.set_vertex(v2)  # Índice 2
    g.set_vertex(v3)  # Índice 3

    print(f"Vértices adicionados. Total de vértices: {g.get_vertex_count()}")
    print(f"É vazio agora? {g.is_empty_graph()}\n")

    # 3. Adição de Arestas
    # Vamos criar um cenário:
    # A(0) -> B(1)
    # B(1) -> C(2)
    # A(0) -> C(2)  (Divergência em A, Convergência em C)
    # C(2) -> D(3)

    print("--- Adicionando Arestas ---")
    g.add_edge(0, 1)  # A -> B
    g.add_edge(1, 2)  # B -> C
    g.add_edge(0, 2)  # A -> C
    g.add_edge(2, 3)  # C -> D

    # Tentativa de adicionar aresta existente (não deve duplicar segundo sua lógica)
    g.add_edge(0, 1)

    print(f"Total de arestas (esperado 4): {g.get_edge_count()}")
    print(f"Existe aresta 0->1 (A->B)? {g.has_edge(0, 1)}")
    print(f"Existe aresta 1->0 (B->A)? {g.has_edge(1, 0)}\n")

    # 4. Teste de Graus (In-degree e Out-degree)
    print("--- Teste de Graus ---")
    # Vértice 2 (C) recebe de B(1) e A(0), e sai para D(3).
    # In-degree esperado: 2, Out-degree esperado: 1
    print(f"Vértice C (idx 2) - Grau de Entrada: {g.get_vertex_in_degree(2)}")
    print(f"Vértice C (idx 2) - Grau de Saída: {g.get_vertex_out_degree(2)}\n")

    # 5. Teste de Pesos
    print("--- Teste de Pesos ---")
    # Pesos de Vértice
    print(f"Peso original de A: {g.get_vertex_weight(0)}")
    g.set_vertex_weight(0, 10.5)
    print(f"Novo peso de A: {g.get_vertex_weight(0)}")

    # Pesos de Aresta
    # Aresta A->B (0->1)
    print(f"Peso aresta 0->1 inicial: {g.get_edge_weight(0, 1)}")
    g.set_edge_weight(0, 1, 5.0)
    print(f"Novo peso aresta 0->1: {g.get_edge_weight(0, 1)}\n")

    # 6. Testes Lógicos (Sucessor, Predecessor, Incidência)
    print("--- Testes Lógicos ---")
    # A(0) -> B(1)
    # B é sucessor de A? Sim (is_sucessor verifica se u é sucessor de v, ou seja v -> u)
    # Na sua implementação: is_sucessor(u, v) retorna has_edge(v, u).
    # Se perguntamos is_sucessor(1, 0) -> has_edge(0, 1) -> True (1 é sucessor de 0)
    print(f"B(1) é sucessor de A(0)? {g.is_sucessor(1, 0)}")

    # A é predecessor de B?
    # is_predessor(u, v) retorna has_edge(u, v).
    # is_predessor(0, 1) -> has_edge(0, 1) -> True.
    print(f"A(0) é predecessor de B(1)? {g.is_predessor(0, 1)}")

    # Incidência: A aresta 0->1 incide em 0? Incide em 1? Incide em 3?
    print(f"Aresta 0->1 incide no vértice 0? {g.is_incident(0, 1, 0)}")
    print(f"Aresta 0->1 incide no vértice 3? {g.is_incident(0, 1, 3)}\n")

    # 7. Teste de Divergência e Convergência
    print("--- Convergência e Divergência ---")

    # Divergência: u1 e u2 devem ser iguais, v1 e v2 diferentes.
    # A(0) vai para B(1) e A(0) vai para C(2).
    # is_divergent(u1, v1, u2, v2)
    print(f"A diverge para B e C? (0->1 e 0->2): {g.is_divergent(0, 1, 0, 2)}")

    # Convergência:
    # Atenção à sua implementação: is_convergent verifica has_edge(v1, u1).
    # Isso sugere que sua lógica inverte a verificação ou espera (destino, origem).
    # Se a lógica for padrão (u1->v1 e u2->v2 onde v1==v2):
    # B(1)->C(2) e A(0)->C(2). Destino comum é C(2).
    # Teste conforme assinatura do método:
    try:
        # u1=1, v1=2 (B->C) e u2=0, v2=2 (A->C)
        result_conv = g.is_convergent(1, 2, 0, 2)
        print(f"B e A convergem para C? {result_conv}")

        # Nota: Se der False, verifique a lógica dentro de graph.py:
        # Se estiver 'has_edge(v1, u1)', o código está checando se C aponta para B, o que seria incomum para convergência.
    except Exception as e:
        print(f"Erro no teste de convergência: {e}")

    print("\n--- Propriedades Globais ---")
    # Conectividade
    # O grafo é conectado? Todos têm alguma aresta, mas vamos ver a lógica is_connected.
    # Sua lógica: Se algum vértice tem in_degree 0 E out_degree 0, retorna False.
    print(f"Grafo é conectado? {g.is_connected()}")

    # Grafo Completo
    print(f"Grafo é completo? {g.is_complete_graph()}")

    print("Gerando arquivo csv")
    g.export_to_gephi("./tables/")

    print("\n=== TESTES FINALIZADOS ===")


if __name__ == "__main__":
    main()
