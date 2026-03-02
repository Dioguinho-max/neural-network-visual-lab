Neural Network Visual Lab
Descrição

Neural Network Visual Lab é um laboratório interativo desenvolvido em JavaScript utilizando a biblioteca p5.js. O projeto implementa do zero uma rede neural do tipo Multilayer Perceptron (MLP) com backpropagation, permitindo visualizar em tempo real o processo de aprendizado ao resolver o problema clássico XOR.

O objetivo é demonstrar, de forma visual e didática, como redes neurais aprendem por meio de ajuste de pesos, propagação direta (forward pass) e retropropagação do erro (backpropagation).

Funcionalidades
1. Implementação completa de rede neural

Arquitetura configurada como 2-3-1:

2 neurônios de entrada

3 neurônios na camada oculta

1 neurônio de saída

Inicialização aleatória de pesos e biases

Atualização dos pesos via Gradiente Descendente

Treinamento online utilizando amostras aleatórias do conjunto XOR

2. Problema XOR

O projeto treina a rede para aprender a função lógica XOR:

Entrada	Saída Esperada
0, 0	0
0, 1	1
1, 0	1
1, 1	0

O XOR é um problema não linearmente separável, exigindo pelo menos uma camada oculta para ser resolvido.

3. Visualização da Rede Neural

O sistema exibe:

Neurônios de entrada, ocultos e saída

Conexões com espessura proporcional ao valor absoluto do peso

Cores diferenciando pesos positivos e negativos

Intensidade dos neurônios baseada na ativação atual

Valores numéricos da saída em tempo real

4. Gráfico de erro

Gráfico dinâmico da evolução do erro

Eixos com escala

Atualização em tempo real

Histórico limitado para melhor visualização

Isso permite observar a convergência do modelo durante o treinamento.

5. Exibição das previsões

O sistema mostra continuamente as previsões da rede para:

[0, 0]

[0, 1]

[1, 0]

[1, 1]

Isso permite acompanhar o quanto o modelo está se aproximando das saídas corretas.

6. Learning Rate em tempo real

Slider interativo

Permite ajustar a taxa de aprendizado dinamicamente

Impacta diretamente na velocidade e estabilidade do treinamento

7. Alternância de função de ativação

O projeto permite alternar entre:

Sigmoid

ReLU

Isso possibilita comparar comportamentos e entender diferenças no aprendizado.
