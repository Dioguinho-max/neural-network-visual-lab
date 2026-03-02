Resumo do Projeto — Neural Network Visual Lab

O Neural Network Visual Lab é uma aplicação web interativa desenvolvida em JavaScript com a biblioteca p5.js, cujo objetivo é demonstrar de forma visual e didática o funcionamento interno de uma rede neural artificial do tipo Multilayer Perceptron (MLP).

O sistema implementa do zero o algoritmo de forward propagation e backpropagation, incluindo:

Inicialização aleatória de pesos e bias

Funções de ativação (Sigmoid e ReLU)

Cálculo do erro

Atualização dos pesos via gradiente descendente

Controle da taxa de aprendizado em tempo real

A rede possui arquitetura 2-3-1 (duas entradas, três neurônios na camada oculta e uma saída) e é treinada para resolver o problema lógico XOR, um caso clássico que não é linearmente separável e exige pelo menos uma camada oculta para ser solucionado.

A aplicação permite:

Visualizar os neurônios e conexões

Observar a variação dos pesos por meio de espessura e cor

Acompanhar a evolução do erro em gráfico com escala

Ver as previsões da rede para todas as combinações de entrada

Alterar dinamicamente a taxa de aprendizado

Comparar o comportamento entre as funções Sigmoid e ReLU

O projeto foi desenvolvido com finalidade educacional, visando consolidar conceitos matemáticos e computacionais fundamentais de redes neurais, além de demonstrar domínio prático da implementação manual do algoritmo de aprendizado supervisionado.

O principal objetivo é evidenciar compreensão técnica dos fundamentos de Machine Learning, indo além do uso de bibliotecas prontas e implementando diretamente os mecanismos internos da rede neural.
