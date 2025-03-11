# Projeto Interdisciplinar - Sistemas de Informação III (2024.2)

## 🎓 **Sobre o Projeto**
Este repositório foi desenvolvido como parte da disciplina de **Projeto Interdisciplinar para Sistemas de Informação III** do **Bacharelado de Sistemas de Informação** na  **Universidade Federal Rural de Pernambuco (UFRPE)**, no semestre **2024.2**.

O projeto consiste em aplicar o método **KDD (Knowledge Discovery in Databases)** para realizar todo o processo de análise de dados e aprendizado de máquina (machine learning), a partir de um **[dataset selecionado no Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)**. Este trabalho envolve etapas de **análise exploratória de dados (AED)**, **pré-processamento**, e a aplicação de **algoritmos de classificação** e **clusterização** para responder a perguntas-chave relacionadas a fatores socioeconômicos e desempenho acadêmico.

O projeto é complementado com um **aplicativo web interativo**, desenvolvido em **Streamlit**, para apresentar os resultados de maneira prática e intuitiva, e um **[artigo científico](https://docs.google.com/document/d/1Wsb2ZdVZy8hvLGr82v3Cs0ESxvbr65g-y7KSkqC_o1w/edit?usp=sharing)** que documenta as questões, objetivos, métodos e resultados obtidos.

Essa disciplina foi feita em conjunto com a disciplina de Desenvolvimento de Sistemas de Informações onde foi desenvolvido uma aplicação mobile e o repositório pode ser acessado [aqui](https://github.com/notsogreatdavi/DSI-2024.2)

### 👥 Equipe do Projeto
- Davi Vieira
- Guilherme Leonardo
- Ronaldo de Araújo
  
**Professor:  Gabriel Alves**

---

## 📋 **Conteúdo do Repositório**
Este repositório contém os seguintes arquivos e diretórios principais:

- `src/`: Scripts em Python com a implementação dos modelos de classificação e clusterização.
- `app/`: Arquivos do Streamlit para o aplicativo web.
- `requirements.txt`: Lista de dependências necessárias para rodar o projeto.
- `data/`: para armazenar os datasets utilizados no projeto.

---

## 👩‍💻 **Como Rodar o Projeto**

Siga os passos abaixo para rodar o projeto em sua máquina:

### 1º - Clone o Repositório
```bash
git clone https://github.com/notsogreatdavi/PISI3-2024.2
cd nome-do-repositorio 
```

### 2º - Crie um Ambiente Virtual 
Com o terminal aberto no diretório do projeto digite o código a seguir:
```bash
python -m venv venv 
```
Este comando é responsável por criar um ambiente virtual, nele todas as bibliotecas necessárias serão instaladas sem interferir diretamente nas bibliotecas da máquina.

### 3º - Acesse o ambiente virtual 
__Caso ocorra erro no próximo comando, vá para o passo 4º e volte para o terceiro passo. Caso não ocorra o erro vá para o passo 5º.__

Ainda com o terminal aberto no diretório do projeto digite o código a seguir:

```bash
.\venv\Scripts\activate
```

Esse comando vai ser responsável por referenciar que agora estamos trabalhando com as dependências do ambiente virutal.

### 4º Passo - Correção de Erro no Acesso ao Ambiente Virtual

Caso seja sua primeira vez acessando e criando um ambiente virtual é necessário alterar a permissão para que isso possa ser feito, por padrão vem desligado.

Vá na barra de pesquisa do Windows e acesse o PowerShell, então digite os comandos: 
```bash
cd /
cd .\Windows\system32
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
__Aparecerá uma confirmação, então digite "s", para confirmar.__

O primeiro código é um comando para mudar o cursor até a raiz do usuário. O segundo é responsável por navegar até a pasta de Administrador, a qual possui a configuração. O terceiro é o código responsável por mudar a configuração padrão assim permitindo criação de ambientes virtuais.

__Caso tenha ocorrido este erro, após essa correção não esqueça de voltar ao passo 3º para acessar o Ambiente Virtual.__

### 5º Passo - Baixar todas as dependências necessárias

Após estar no ambiente virtual (3º passo), escreva no mesmo terminal o seguinte comando:

```
pip install - r requirements.txt
```

Este comando é responsável por baixar todas as bibliotecas que serão usadas no projeto, e todas estão descritas no arquivo. 

### 6º Passo - Acessar o streamlit
Com o ambiente virtual acessado e os requirements baixados é possível agora rodar o streamlit para visualização.


Escreva no terminal o seguinte comando: 
```
streamlit run '.\Home.py'
```

### Obrigado pela atenção 🤝
