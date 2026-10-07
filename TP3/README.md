# TP3: guia de uso do código

Este guia explica como usar os programas deste diretório para fazer as
medições do TP3 e gerar os gráficos do relatório.

## A ideia geral

São quatro peças, usadas nesta ordem:

```
 Arduino                 computador
┌──────────────┐  USB   ┌──────────────┐     ┌──────────┐     ┌──────────────┐
│  firmware    │ ─────> │  coleta.py   │ ──> │  dados/  │ ──> │  analise.py  │ ──> gráficos e números
│ (lê o sensor)│        │ (grava tudo) │     │  (CSVs)  │     │  (faz contas)│     para o relatório
└──────────────┘        └──────────────┘     └──────────┘     └──────────────┘
```

1. **Firmware** (programa que roda no Arduino). Acende e apaga o LED, lê
   o sensor e manda os números pela USB. Há dois:
   - `firmware/aquisicao`: usado em quase todo o laboratório, para medir;
   - `firmware/distancia`: usado só no final, já mostra a distância em mm.
2. **`scripts/coleta.py`**. Conversa com o Arduino, pergunta a distância
   em que o cartão está e salva as leituras num arquivo CSV em `dados/`.
   Ninguém precisa anotar números à mão.
3. **`scripts/analise.py`**. Lê todos os CSVs, faz as contas (médias,
   ajuste do modelo, alcance) e gera os gráficos em `figuras/`.
4. **`TP3_planilha.ods`**. Faz as mesmas contas em planilha, porque o
   roteiro pede uma planilha. Basta colar os CSVs nela.

---

## Antes de ir ao laboratório

Instalar no notebook que vai para o laboratório:

```
pip install pyserial numpy matplotlib
```

e o PlatformIO (ou usar o Arduino IDE, copiando o `src/main.cpp` de cada
firmware para um sketch).

Todos os comandos deste guia são digitados **dentro da pasta `TP3/`**.

Duas coisas que causam erro com frequência:

- **A porta.** Com o Arduino ligado na USB, `pio device list` mostra o
  nome da porta. No Linux costuma ser `/dev/ttyACM0`, que é o nome usado
  nos exemplos abaixo. Se for outro, troque nos comandos.
- **Só um programa por vez na porta.** Se o monitor serial ou o
  `plot_serial.py` estiver aberto, o `coleta.py` não consegue abrir a
  porta. Feche um antes de abrir o outro.

---

## No laboratório, passo a passo

Tempo total: cerca de 2 h 30 min. A parte longa é o passo 3, porque cada
leitura com o LED em 1 s leva 2 s.

### Passo 1: montar e conferir o circuito

Montagem do roteiro: D8 → resistor de 100 Ω → anodo do LED, catodo no
GND; 5 V → resistor de 4,7 kΩ → coletor do fototransistor → A0, emissor no
GND. Com a face preta virada para você e os pinos para baixo, os
terminais são: coletor, emissor, anodo, catodo.

1. Com o multímetro, medir os dois resistores (fora do circuito), a
   tensão de 5 V na protoboard e a tensão sobre o LED aceso.
2. Gravar o firmware de medição:
   ```
   pio run -d firmware/aquisicao -t upload
   ```
3. Abrir o monitor serial (`pio device monitor -b 115200`). A cada 2 s
   aparece uma linha assim:
   ```
   A,1000000,1000000,412,1008,596
   ```
   Os três últimos números são: leitura com o LED aceso, leitura com o
   LED apagado e o sinal (a diferença entre as duas).
4. Conferir:
   - sensor todo tampado com algo escuro: o 2º desses números (LED
     apagado) fica perto de **1023**. Essa é a dica do professor. Se der
     bem menos (ex.: 900), conferir o resistor de 4,7 kΩ. Os programas
     continuam funcionando, porque usam a diferença entre as leituras,
     mas a faixa útil do sinal fica menor;
   - cartão branco encostado no sensor: o 1º número (LED aceso) cai
     bastante e para de mudar. Ele **não precisa chegar a 0**: o sensor
     saturado costuma parar entre ~40 e ~80. Anotar esse valor;
   - nada na frente do sensor: o sinal (último número) fica perto de 0.
5. Ver o LED pela câmera do celular (aparece uma luz roxa).
6. Anotar tudo na aba **Parametros** da planilha. Fotografar a
   protoboard e a bancada com a régua e o cartão.
7. **Fechar o monitor serial.**

### Passo 2: medir o ruído sem cartão ("fundo")

**Para que serve:** saber quanto o sinal oscila quando não há nada na
frente do sensor. É o que define até onde o sensor "enxerga": o cartão
só é detectado se o sinal ficar claramente acima desse ruído.

Tirar o cartão da frente do sensor e rodar:

```
python3 scripts/coleta.py serie /dev/ttyACM0 --serie fundo --saida dados/fundo.csv --sem-distancia --n 100 --tempos-ms 1000,100,10,1,0.5,0.2
```

O programa pede um Enter e mede 100 leituras para cada tempo de LED da
lista. Não é preciso fazer mais nada até ele terminar (~4 min).

### Passo 3: curva sinal × distância, de perto para longe ("ida")

**Para que serve:** é a medição principal. Com ela se ajusta o modelo
que converte o sinal em milímetros.

```
python3 scripts/coleta.py serie /dev/ttyACM0 --serie ida --saida dados/ida.csv
```

O programa sugere a próxima distância e espera. A tela fica assim:

```
d (mm) [0]:                       <- cartão em 0 mm, apertar Enter
  0 mm, t_on = 1000 ms, t_off = 1000 ms
    20/20  adc_on=   3 adc_off=1010 sinal=1007
    S = 1006.8 ± 1.9   adc_on médio = 3

d (mm) [2]:                       <- mover o cartão para 2 mm, Enter
  ...
```

Em cada posição:

| Você digita | O que acontece |
|---|---|
| Enter | mede na distância sugerida entre colchetes |
| um número, ex. `37` | mede em 37 mm em vez da sugerida |
| `r` | refaz a última posição (as leituras antigas dela são apagadas) |
| `q` | termina |

As distâncias sugeridas vão de 2 em 2 mm até 20 mm, de 5 em 5 até 100,
de 10 em 10 até 200 e de 20 em 20 até 300. Cada posição leva ~45 s. Se em
300 mm o sinal ainda estiver claramente acima de zero, continue (ele
segue sugerindo de 20 em 20 mm). O arquivo é salvo a cada posição, então
dá para interromper e retomar depois sem perder nada.

### Passo 4: a mesma curva, de longe para perto ("volta")

**Para que serve:** conferir o modelo com medidas que não foram usadas
para ajustá-lo. É daqui que sai o erro em mm do relatório.

```
python3 scripts/coleta.py serie /dev/ttyACM0 --serie volta --saida dados/volta.csv --reverso
```

Funciona igual ao passo 3, mas as sugestões começam pela distância mais
longe.

### Passo 5: primeira olhada nos resultados

```
python3 scripts/analise.py dados
```

Aparece na tela (e fica salvo em `resultados.txt`) um resumo como este:

```
fundo: N0 = 100, mu0 = 0.94, sigma0 = 1.48, S_lim = 5.37 contagens
alcance de detecção d_det = 150 mm
200 mm dentro do alcance de detecção? não
M1: g0 = -0.255 mm, g1 = 385.279 mm·cont^0.5, R² = 0.9978
```

- `g0` e `g1` são os ganhos do modelo `d = g0 + g1/√S` do roteiro;
- `d_det` é até onde o sensor detecta o cartão;
- se aparecer "ATENÇÃO: medir mais longe", o passo 3 parou cedo demais.

Com isso, escolher três distâncias com sinal bem claro (por exemplo 20,
60 e 120 mm) para o passo 7.

### Passo 6: tempo de resposta do sensor ("degrau")

**Para que serve:** medir quanto tempo o sensor leva para reagir quando
o LED acende e apaga. É o que explica, no relatório, por que tempos
curtos de LED dão menos sinal.

Cartão parado a 30 mm:

```
python3 scripts/coleta.py degrau /dev/ttyACM0 --saida dados/degrau_30mm.csv --rapido
```

Depois repetir com o cartão a 5 mm, salvando em
`dados/degrau_5mm.csv`. Cada medida leva poucos segundos.

Se houver osciloscópio, é ainda melhor: ponta 1 no D8, ponta 2 no A0, e
medir o tempo que o A0 leva de 10 % a 90 % da variação. Anotar na aba
**Degrau** da planilha.

### Passo 7: o que acontece com tempos menores de LED

**Para que serve:** o professor pediu para explorar o tempo em que o LED
fica aceso. São três medições, todas com o mesmo comando, mudando as
opções.

**7a. Mesmo tempo aceso e apagado**, de 100 ms até 0,2 ms, nas três
distâncias escolhidas no passo 5. O programa mede todos os tempos antes
de pedir para mover o cartão:

```
python3 scripts/coleta.py serie /dev/ttyACM0 --serie T1 --saida dados/T1.csv --grade 20,60,120 --tempos-ms 100,10,1,0.5,0.2 --n 50
```

**7b. Mudando só o tempo aceso, ou só o tempo apagado**, com o cartão a
60 mm. Mostra qual dos dois limita o sinal:

```
python3 scripts/coleta.py serie /dev/ttyACM0 --serie T2on --saida dados/T2on.csv --grade 60 --ton-ms 100,10,5,2,1,0.5,0.2,0.1 --toff-ms 10 --n 50
python3 scripts/coleta.py serie /dev/ttyACM0 --serie T2off --saida dados/T2off.csv --grade 60 --ton-ms 10 --toff-ms 100,10,5,2,1,0.5,0.2,0.1 --n 50
```

**7c. Alcance para cada tempo**: o cartão de 20 em 20 mm, até 400 mm,
medido com 10, 1 e 0,2 ms:

```
python3 scripts/coleta.py serie /dev/ttyACM0 --serie T3 --saida dados/T3.csv --grade 20:400:20 --tempos-ms 10,1,0.2
```

O tempo de 1000 ms não precisa ser medido de novo em 7a e 7c: o
`analise.py` usa os dados do passo 3.

### Passo 8: o Arduino mostrando a distância em mm

**Para que serve:** é o desafio do roteiro: o próprio Arduino calcular a
distância.

1. Gerar o arquivo com os ganhos do modelo e gravar o segundo firmware:
   ```
   python3 scripts/analise.py dados --header firmware/distancia/include/calibracao.h
   pio run -d firmware/distancia -t upload
   ```
2. Ver a distância ao vivo, mexendo o cartão, e **tirar print da tela**
   para o relatório:
   ```
   python3 exemplos/plot_serial.py /dev/ttyACM0
   ```
3. Fechar o gráfico e conferir contra a régua em ~10 posições que não
   estejam na lista do passo 3 (ex.: 15, 33, 47, 72 mm):
   ```
   python3 scripts/coleta.py vivo /dev/ttyACM0 --saida dados/vivo.csv
   ```
   O programa pede a distância da régua, mostra a média do Arduino e o
   erro. Enter vazio termina.

### Passo 9: antes de sair

Desligar tudo, inclusive os ventiladores da sala 2010.

---

## Depois do laboratório

1. **Gráficos e números finais:**
   ```
   python3 scripts/analise.py dados
   ```
   Gera os gráficos em `figuras/` e o resumo em `resultados.txt`. Para
   colocar um gráfico no relatório, trocar a caixa cinza `\placeholderfig`
   correspondente por `\includegraphics[width=\linewidth]{nome}`.

   | Gráfico (`figuras/`) | O que mostra |
   |---|---|
   | `sinal_dist` | sinal × distância (ida e volta), com o ruído de fundo e o alcance |
   | `sinal_dist_log` | o mesmo, em escala log: mostra onde o sinal encontra o fundo |
   | `adc_on_off` | leituras com LED aceso e apagado: mostra onde o sensor satura |
   | `ruido` | quanto o sinal oscila em cada distância |
   | `dest_dist` | distância calculada × distância da régua, para os três modelos |
   | `erro` | erro em mm de cada modelo |
   | `resolucao` | incerteza da distância em mm, que cresce com a distância |
   | `ao_vivo` | o que o Arduino mostrou × a régua |
   | `degrau` | resposta do sensor quando o LED acende e apaga |
   | `sinal_tempo` | quanto o sinal cai com tempos menores de LED (7a) |
   | `ton_toff` | efeito separado do tempo aceso e do tempo apagado (7b) |
   | `alcance_tempo` | alcance para cada tempo de LED (7c) |

2. **Planilha:** abrir cada CSV de `dados/` (menos os de degrau e vivo)
   no LibreOffice e escolher o idioma **Inglês (EUA)** na janela de
   importação, porque os arquivos usam ponto como separador decimal.
   Copiar as linhas, sem o cabeçalho, para a aba **Bruto**, um arquivo
   embaixo do outro, nas colunas A a H. O resto da planilha se calcula
   sozinho. Os números devem bater com os do `analise.py`.

---

## Referência

Esta parte só é necessária para entender os detalhes ou mudar algo.

### Os três modelos comparados

| Modelo | Fórmula | Ideia |
|---|---|---|
| M1 | d = g0 + g1/√S | o do roteiro: o sinal cai com o quadrado da distância |
| M2 | S = α·d^(−n) | o mesmo, mas o expoente n é medido em vez de supor 2 |
| M3 | tabela | guarda os pares (sinal, distância) medidos e interpola entre eles |

### Regras usadas nas contas

As mesmas no `analise.py` e na planilha:

- **saturado:** com o cartão muito perto, o sensor recebe luz demais e a
  leitura com LED aceso para num patamar (não em 0, e sim em ~40–80,
  dependendo do resistor). Esses pontos ficam fora do ajuste. O patamar
  é **medido nos dados**: o menor valor da ida + 5. Se esse menor valor
  passar de 200, considera-se que o sensor não saturou. Por isso nada
  depende do valor exato dos resistores;
- **limiar de detecção:** média do fundo + 3 desvios-padrão;
- **alcance de detecção:** a maior distância em que o sinal ainda fica
  acima do limiar;
- **alcance de medição:** a maior distância em que a incerteza da
  distância fica abaixo de 5 % dela;
- **ajuste:** usa os pontos da ida entre o pico do sinal e o alcance de
  detecção; a **validação** usa os pontos da volta nessa mesma faixa.

### Firmware de medição (`firmware/aquisicao`)

Faz o ciclo do roteiro (acende o LED, espera, lê; apaga, espera, lê) e
imprime uma linha por ciclo:

```
A, tempo aceso (µs), tempo apagado (µs), leitura LED aceso, leitura LED apagado, sinal
```

Diferenças em relação ao sketch do roteiro:

- cada leitura é a média de 8 leituras seguidas, o que reduz o ruído.
  Com tempos abaixo de 5 ms usa uma só, para não alterar o tempo
  estudado;
- os tempos do LED mudam por comando, sem gravar o programa de novo. O
  `coleta.py` faz isso sozinho;
- com tempos curtos, mede 100 ciclos seguidos e só depois envia. Enviar
  uma linha leva ~2 ms e, se fosse enviada a cada ciclo, o LED ficaria
  apagado mais tempo que o pedido.

Comandos que podem ser digitados no monitor serial:

| Comando | Efeito |
|---|---|
| `on 1000000` | tempo aceso, em microssegundos (1000000 = 1 s) |
| `off 1000000` | tempo apagado, em microssegundos |
| `m 8` | quantas leituras entram em cada média |
| `pausa` / `go` | para / volta a medir |
| `?` | mostra a configuração atual |

O comando só é aceito no fim do ciclo em andamento (até 2 s com o tempo
de 1 s).

### Firmware de distância (`firmware/distancia`)

Imprime `sinal,mm`. Os ganhos ficam em `include/calibracao.h`, escrito
pelo `analise.py --header` (o que está no repositório tem valores
provisórios). Quando o sinal está no nível do ruído, imprime o alcance
máximo ("está a essa distância ou mais"). Quando o sensor satura, imprime
a menor distância do ajuste. `#define MODELO 1`, `2` ou `3` escolhe o
modelo. Ao ligar, imprime quanto tempo cada modelo leva para calcular.

### Opções do `coleta.py serie`

| Opção | Padrão | Para que serve |
|---|---|---|
| `--serie` | (obrigatória) | nome da medição: `fundo`, `ida`, `volta`, `T1`, `T2on`, `T2off`, `T3` |
| `--saida` | (obrigatória) | arquivo CSV onde salvar |
| `--n` | 20 | leituras por posição |
| `--descarta` | 2 | leituras ignoradas no começo de cada posição (o cartão pode ainda estar se movendo) |
| `--grade` | 0 a 300 mm | distâncias sugeridas; `20:400:20` = de 20 a 400, de 20 em 20 |
| `--reverso` | | sugere as distâncias de longe para perto |
| `--sem-distancia` | | medição sem cartão |
| `--tempos-ms` | | lista de tempos de LED, iguais aceso e apagado |
| `--ton-ms`, `--toff-ms` | 1000 | tempos aceso e apagado separados |

Outros modos: `coleta.py degrau` (opções `--k` amostras, `--rep`
repetições, `--rapido`) e `coleta.py vivo`. `python3 scripts/coleta.py
serie -h` lista tudo.

### Opções do `analise.py`

| Opção | Padrão | Para que serve |
|---|---|---|
| `--figs` | `figuras` | pasta dos gráficos |
| `--fmt` | `pdf` | `pdf` ou `png` |
| `--descartar` | | distâncias a tirar do ajuste, ex.: `14,16` (justificar no relatório) |
| `--header` | | onde escrever o `calibracao.h` |
| `--modelo` | 1 | modelo usado no `calibracao.h` |
| `--k`, `--res` | 3, 0,05 | limiar de detecção e resolução (mudar só se mudar também na planilha) |
| `--sat-margem`, `--sat-max` | 5, 200 | regra da saturação (ver acima) |
| `--sat` | | força um limiar fixo de saturação, se a detecção automática errar |

### Arquivos

```
TP3/
├── firmware/aquisicao/     firmware de medição
├── firmware/distancia/     firmware que mostra mm (ganhos em include/calibracao.h)
├── scripts/coleta.py       grava as medições em dados/
├── scripts/analise.py      contas e gráficos
├── exemplos/               material original do roteiro (inclui plot_serial.py)
├── dados/                  CSVs das medições
├── figuras/                gráficos gerados
├── TP3_planilha.ods        planilha
└── main.tex                relatório
```
