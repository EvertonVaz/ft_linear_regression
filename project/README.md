# 42 São Paulo - ft_linear_regression
> **Sobre o projeto:**
> Este projeto é uma implementação do zero de um algoritmo de Regressão Linear Simples, sem o uso de bibliotecas de aprendizado de máquina.<br><br>
> O objetivo é prever o preço de carros com base em sua quilometragem, estabelecendo uma relação linear entre a quilometragem (variável independente) e o preço (variável dependente), utilizando a hipótese `estimatePrice(mileage) = θ0 + (θ1 * mileage)`.<br>
> O modelo é treinado usando o algoritmo de Descida de Gradiente, que ajusta os parâmetros θ0 e θ1 iterativamente para minimizar o erro (Função de Custo - MSE). A normalização dos dados é um passo crucial antes do treinamento para otimizar o processo de convergência.
>
> O projeto inclui dois programas principais e um bonus:
> - **training.py:** Lê o conjunto de dados, treina o modelo usando a descida de gradiente e salva os parâmetros (θ0, θ1, médias e desvios padrão) em um arquivo JSON.
> - **estimatePrice.py:** Carrega os parâmetros salvos para fazer previsões sobre novas quilometragens de entrada.
> - **bonus.py:** Implementa o plot de regressão linear, visualizando a relação entre quilometragem e preço, além de exibir a linha de regressão.

# Dependencias
1. Docker e Docker Compose
1. Python e poetry (caso queira rodar localmente)
    > Eu recomendo o uso do Docker para evitar problemas de compatibilidade entre versões de bibliotecas e Python.<br>
    > O poetry esta configurado (poetry.toml) para não criar uma virtualenv, pois o container já isola o ambiente.<br>


# Execução:
- Clonar o repositório:
```bash
git clone git@github.com:EvertonVaz/ft_linear_regression.git
```

- Subir o container:
```bash
docker-compose up --build -d
```
- Apos subir o container você pode acessar-lo pelo devcontainer do VSCode ou pelo terminal:
    - devcontainer do VSCode:
        ```
        Pressione F1 e selecione "Reopen in Container" ou "Reopen in Dev Container"
        ```
    - terminal:
        ```bash
        docker-compose exec dev zsh
        ```

- Rodar o projeto:
```bash
cd project && poetry install
task test # para rodar os testes
task estimate # para rodar o estimador
task train # para treinar o modelo
task bonus # para rodar o bonus
```

- Parar o container e remover imagens e volumes:
```bash
docker-compose down --rmi all -v
```