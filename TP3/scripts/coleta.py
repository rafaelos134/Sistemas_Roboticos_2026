#!/usr/bin/env python3
"""Coleta as medidas do TCRT5000 pela serial e grava CSV (TP3).

Três modos, um por tipo de ensaio:

  serie   ciclos do firmware de aquisição, por distância e tempo
          -> serie,t_on_us,t_off_us,d_mm,rep,adc_on,adc_off,sinal
  degrau  resposta do A0 logo após o LED ligar e desligar
          -> t_us,a0_liga,a0_apaga,t_us_apaga
  vivo    leituras sinal,mm do firmware de distância, por posição da régua
          -> d_regua_mm,rep,sinal,mm

Exemplos (rodar da pasta TP3):

  python3 scripts/coleta.py serie /dev/ttyACM0 --serie ida   --saida dados/ida.csv
  python3 scripts/coleta.py serie /dev/ttyACM0 --serie volta --saida dados/volta.csv --reverso
  python3 scripts/coleta.py serie /dev/ttyACM0 --serie fundo --saida dados/fundo.csv \\
      --sem-distancia --n 100
  python3 scripts/coleta.py serie /dev/ttyACM0 --serie T1 --saida dados/T1.csv \\
      --grade 20,60,120 --tempos-ms 1000,100,10,1,0.5,0.2 --n 50
  python3 scripts/coleta.py degrau /dev/ttyACM0 --saida dados/degrau_30mm.csv --rapido
  python3 scripts/coleta.py vivo /dev/ttyACM0 --saida dados/vivo.csv

Detalhes de cada modo no README.md.
"""

import argparse
import csv
import itertools
import os
import sys
import time

import serial

CAMPOS_SERIE = ["serie", "t_on_us", "t_off_us", "d_mm", "rep", "adc_on", "adc_off", "sinal"]
CAMPOS_DEGRAU = ["t_us", "a0_liga", "a0_apaga", "t_us_apaga"]
CAMPOS_VIVO = ["d_regua_mm", "rep", "sinal", "mm"]

# Grade da metodologia (Seção 4.3 do relatório): mais densa perto do sensor.
GRADE_PADRAO = "0:20:2,25:100:5,110:200:10,220:300:20"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def numeros(texto):
    """'1000,100,0.5' -> [1000.0, 100.0, 0.5]"""
    return [float(x) for x in texto.split(",") if x.strip()]


def expande_grade(texto):
    """'0:20:2,25,30' -> [0, 2, ..., 20, 25, 30]. Intervalos incluem o fim."""
    saida = []
    for parte in texto.split(","):
        parte = parte.strip()
        if not parte:
            continue
        if ":" in parte:
            a, b, passo = (float(x) for x in parte.split(":"))
            k = 0
            while a + k * passo <= b + 1e-9:
                saida.append(round(a + k * passo, 3))
                k += 1
        else:
            saida.append(float(parte))
    return saida


def fmt_num(x):
    """Inteiro sem '.0'; senão, o float como está."""
    if x is None or x == "":
        return ""
    x = float(x)
    return str(int(x)) if x.is_integer() else repr(x)


def ms_para_us(ms):
    return int(round(ms * 1000))


def le_csv(caminho, campos):
    if not os.path.exists(caminho):
        return []
    with open(caminho, newline="") as f:
        leitor = csv.DictReader(f)
        if leitor.fieldnames != campos:
            sys.exit(f"{caminho}: colunas {leitor.fieldnames}, esperado {campos}. Use outro arquivo.")
        return list(leitor)


def grava_csv(caminho, campos, linhas):
    os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)
    tmp = caminho + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    os.replace(tmp, caminho)


class Porta:
    """Serial com leitura de linha e espera por resposta."""

    def __init__(self, nome):
        try:
            self.ser = serial.Serial(nome, 115200, timeout=0.5)
        except serial.SerialException as e:
            from serial.tools import list_ports
            portas = [p.device for p in list_ports.comports() if "USB" in p.device or "ACM" in p.device]
            if not os.path.exists(nome):
                dica = (f"a porta {nome} não existe. Portas encontradas agora: {', '.join(portas) or 'nenhuma'}"
                        " (o nome muda quando o cabo é reconectado)")
            else:
                dica = ("a porta existe mas está ocupada: feche o monitor serial (Ctrl+C no terminal dele)"
                        " ou o plot_serial.py")
            sys.exit(f"não abriu {nome}: {e}\n{dica}")
        time.sleep(2)  # a Mega reinicia ao abrir a USB
        self.ser.reset_input_buffer()
        self.config = None

    def linha(self, timeout):
        fim = time.time() + timeout
        while time.time() < fim:
            bruto = self.ser.readline()
            if bruto:
                return bruto.decode(errors="ignore").strip()
        return None

    def envia(self, cmd, espera=None, timeout=5.0):
        """Envia um comando e, se pedido, espera uma linha que começa com `espera`."""
        self.ser.write((cmd + "\n").encode())
        if espera is None:
            return None
        fim = time.time() + timeout
        while time.time() < fim:
            l = self.linha(fim - time.time())
            if l is None:
                break
            if l.startswith("# erro"):
                sys.exit(f"firmware recusou '{cmd}': {l}")
            if l.startswith(espera):
                return l
        sys.exit(f"sem resposta a '{cmd}' em {timeout:.0f} s. O firmware gravado é o de aquisição?")

    def fecha(self):
        self.ser.close()


# ---------------------------------------------------------------------------
# Modo serie
# ---------------------------------------------------------------------------
def pares_de_tempo(args):
    if args.tempos_ms:
        return [(t, t) for t in numeros(args.tempos_ms)]
    return list(itertools.product(numeros(args.ton_ms), numeros(args.toff_ms)))


def configura(porta, t_on_us, t_off_us, m):
    # Os comandos só são lidos entre ciclos (até ~2 s com 1 s + 1 s), então
    # vão os três juntos e espera-se só a confirmação do último.
    if porta.config == (t_on_us, t_off_us, m):
        return
    porta.envia(f"m {m}")
    porta.envia(f"on {t_on_us}")
    porta.envia(f"off {t_off_us}", "#ok off", timeout=10)
    porta.config = (t_on_us, t_off_us, m)


def coleta_ciclos(porta, t_on_us, t_off_us, n, descarta):
    """Lê ciclos com os tempos pedidos; descarta os primeiros e devolve n."""
    duracao = (t_on_us + t_off_us) / 1e6
    timeout = max(5.0, 4 * duracao)
    # Enquanto o programa esperava a próxima distância, o Arduino continuou
    # medindo, e essas linhas ficaram acumuladas na porta. Esvazia o buffer e
    # pede a configuração: o firmware só responde no fim do ciclo (ou bloco)
    # em andamento, então tudo o que chega depois da resposta foi medido
    # depois do Enter.
    porta.ser.reset_input_buffer()
    porta.envia("?", "# t_on_us=", timeout=max(15.0, 4 * duracao))
    validos = []
    ignorados = 0
    while len(validos) < n:
        l = porta.linha(timeout)
        if l is None:
            sys.exit("a serial parou de enviar ciclos. Conferir a placa e o cabo.")
        if not l.startswith("A,"):
            continue
        partes = l.split(",")
        if len(partes) != 6:
            continue
        _, ton, toff, on, off, s = partes
        if int(ton) != t_on_us or int(toff) != t_off_us:
            continue  # ciclo ainda com os tempos antigos
        if ignorados < descarta:
            ignorados += 1
            continue
        validos.append((int(on), int(off), int(s)))
        if duracao > 0.2:
            print(f"\r    {len(validos)}/{n}  adc_on={on:>4} adc_off={off:>4} sinal={s:>4}", end="", flush=True)
    if duracao > 0.2:
        print()
    return validos


def resumo(validos):
    s = [v[2] for v in validos]
    on = [v[0] for v in validos]
    media = sum(s) / len(s)
    dp = (sum((x - media) ** 2 for x in s) / (len(s) - 1)) ** 0.5 if len(s) > 1 else 0.0
    return f"S = {media:.1f} ± {dp:.1f}   adc_on médio = {sum(on) / len(on):.0f}"


def modo_serie(args):
    linhas = le_csv(args.saida, CAMPOS_SERIE)
    tempos = pares_de_tempo(args)
    porta = Porta(args.porta)
    porta.envia("go")

    if args.sem_distancia:
        distancias = [None]
    else:
        distancias = expande_grade(args.grade)
        if args.reverso:
            distancias = distancias[::-1]

    print(f"série '{args.serie}', {len(tempos)} par(es) de tempo, N = {args.n}, descarta {args.descarta}")
    print(f"arquivo: {args.saida} ({len(linhas)} linhas já existentes)")
    if not args.sem_distancia:
        print("Em cada posição: Enter aceita a distância sugerida, um número usa outra,")
        print("'r' repete a última posição, 'q' encerra. Repetir uma distância substitui as linhas dela.")

    i = 0
    ultima = None
    while True:
        if args.sem_distancia:
            if i > 0:
                break
            input("Sem cartão na frente do sensor. Enter para começar...")
            d = None
            i = 1
        else:
            sugestao = distancias[i] if i < len(distancias) else (ultima + 20 if ultima is not None else 0)
            resp = input(f"\nd (mm) [{fmt_num(sugestao)}]: ").strip().lower()
            if resp == "q":
                break
            if resp == "r":
                if ultima is None:
                    continue
                d = ultima
            elif resp == "":
                d = sugestao
                i += 1
            else:
                try:
                    d = float(resp.replace(",", "."))
                except ValueError:
                    print("  entrada inválida")
                    continue
        for t_on_ms, t_off_ms in tempos:
            t_on_us, t_off_us = ms_para_us(t_on_ms), ms_para_us(t_off_ms)
            configura(porta, t_on_us, t_off_us, args.m)
            rotulo = "fundo" if d is None else f"{fmt_num(d)} mm"
            print(f"  {rotulo}, t_on = {fmt_num(t_on_ms)} ms, t_off = {fmt_num(t_off_ms)} ms")
            validos = coleta_ciclos(porta, t_on_us, t_off_us, args.n, args.descarta)
            chave = (args.serie, str(t_on_us), str(t_off_us), fmt_num(d))
            antes = len(linhas)
            linhas = [l for l in linhas
                      if (str(l["serie"]), str(l["t_on_us"]), str(l["t_off_us"]), str(l["d_mm"])) != chave]
            if len(linhas) < antes:
                print(f"    substituídas {antes - len(linhas)} linhas antigas desta posição")
            for rep, (on, off, s) in enumerate(validos, 1):
                linhas.append({"serie": args.serie, "t_on_us": t_on_us, "t_off_us": t_off_us,
                               "d_mm": fmt_num(d), "rep": rep, "adc_on": on, "adc_off": off, "sinal": s})
            grava_csv(args.saida, CAMPOS_SERIE, linhas)
            print(f"    {resumo(validos)}")
        ultima = d

    porta.fecha()
    print(f"\n{len(linhas)} linhas em {args.saida}")


# ---------------------------------------------------------------------------
# Modo degrau
# ---------------------------------------------------------------------------
def modo_degrau(args):
    porta = Porta(args.porta)
    porta.envia("pausa", "#ok pausa", timeout=10)
    porta.envia("adc rapido" if args.rapido else "adc normal", "#ok adc")
    input(f"Cartão na posição do ensaio. Enter para medir ({args.rep} repetições)...")
    porta.envia(f"degrau {args.k} {args.rep}", "#ok degrau")
    curvas = {0: [], 1: []}
    while True:
        l = porta.linha(30)
        if l is None:
            sys.exit("o degrau não terminou. Conferir o firmware.")
        if l.startswith("#fim degrau"):
            break
        if l.startswith("G,"):
            _, borda, t, a0 = l.split(",")
            curvas[int(borda)].append((int(t), float(a0)))
    porta.envia("adc normal", "#ok adc")
    porta.envia("go", "#ok go")
    porta.fecha()

    liga, apaga = curvas[1], curvas[0]
    linhas = [{"t_us": tl, "a0_liga": al, "a0_apaga": aa, "t_us_apaga": ta}
              for (tl, al), (ta, aa) in zip(liga, apaga)]
    grava_csv(args.saida, CAMPOS_DEGRAU, linhas)
    print(f"{len(linhas)} amostras por borda em {args.saida}")
    print(f"A0 ao ligar: {liga[0][1]:.0f} -> {liga[-1][1]:.0f};  ao apagar: {apaga[0][1]:.0f} -> {apaga[-1][1]:.0f}")
    print(f"janela: {liga[-1][0]} us. Se a curva não estabilizou, aumentar --k ou tirar --rapido.")


# ---------------------------------------------------------------------------
# Modo vivo
# ---------------------------------------------------------------------------
def modo_vivo(args):
    linhas = le_csv(args.saida, CAMPOS_VIVO)
    porta = Porta(args.porta)
    print("Firmware de distância (sinal,mm). Enter vazio encerra.")
    while True:
        resp = input("\nd da régua (mm): ").strip()
        if not resp:
            break
        try:
            d = float(resp.replace(",", "."))
        except ValueError:
            print("  entrada inválida")
            continue
        porta.ser.reset_input_buffer()
        mm = []
        ignorados = 0
        while len(mm) < args.n:
            l = porta.linha(10)
            if l is None:
                sys.exit("a serial parou. O firmware gravado é o de distância?")
            partes = l.split(",")
            if len(partes) != 2 or not partes[0].isdigit():
                continue
            if ignorados < args.descarta:
                ignorados += 1
                continue
            mm.append((int(partes[0]), float(partes[1])))
            print(f"\r    {len(mm)}/{args.n}  {partes[1]} mm", end="", flush=True)
        print()
        linhas = [l for l in linhas if str(l["d_regua_mm"]) != fmt_num(d)]
        for rep, (s, x) in enumerate(mm, 1):
            linhas.append({"d_regua_mm": fmt_num(d), "rep": rep, "sinal": s, "mm": x})
        grava_csv(args.saida, CAMPOS_VIVO, linhas)
        vals = [x for _, x in mm]
        media = sum(vals) / len(vals)
        print(f"    média = {media:.1f} mm, erro = {media - d:+.1f} mm")
        # Linha pronta para colar na aba Arduino_ao_vivo (separada por tabulação).
        print("    planilha: " + "\t".join([fmt_num(d)] + [f"{x:.1f}" for x in vals]))
    porta.fecha()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="modo", required=True)

    s = sub.add_parser("serie", help="ciclos do firmware de aquisição")
    s.add_argument("porta")
    s.add_argument("--serie", required=True, help="ida, volta, fundo, T1, T2on, T2off, T3")
    s.add_argument("--saida", required=True, help="CSV de saída (acrescenta/substitui linhas)")
    s.add_argument("--n", type=int, default=20, help="ciclos válidos por ponto (padrão 20)")
    s.add_argument("--descarta", type=int, default=2, help="ciclos descartados após mudar de ponto (padrão 2)")
    s.add_argument("--m", type=int, default=8, help="conversões por leitura (o firmware usa 1 abaixo de 5 ms)")
    s.add_argument("--grade", default=GRADE_PADRAO, help=f"distâncias em mm, 'ini:fim:passo,...' (padrão {GRADE_PADRAO})")
    s.add_argument("--reverso", action="store_true", help="percorre a grade de longe para perto (volta)")
    s.add_argument("--sem-distancia", action="store_true", help="série sem cartão (fundo)")
    s.add_argument("--tempos-ms", help="lista de tempos com t_on = t_off, ex.: 1000,100,10,1")
    s.add_argument("--ton-ms", default="1000", help="lista de t_on (ms); combinada com --toff-ms")
    s.add_argument("--toff-ms", default="1000", help="lista de t_off (ms)")
    s.set_defaults(func=modo_serie)

    g = sub.add_parser("degrau", help="resposta ao degrau do LED")
    g.add_argument("porta")
    g.add_argument("--saida", required=True)
    g.add_argument("--k", type=int, default=200, help="amostras após cada borda (máx. 250)")
    g.add_argument("--rep", type=int, default=20, help="repetições promediadas (máx. 50)")
    g.add_argument("--rapido", action="store_true", help="ADC com divisor 16 (~16 us por amostra)")
    g.set_defaults(func=modo_degrau)

    v = sub.add_parser("vivo", help="validação do firmware de distância contra a régua")
    v.add_argument("porta")
    v.add_argument("--saida", required=True)
    v.add_argument("--n", type=int, default=20)
    v.add_argument("--descarta", type=int, default=2)
    v.set_defaults(func=modo_vivo)

    args = ap.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\ninterrompido; o que já foi gravado está no CSV.")


if __name__ == "__main__":
    main()
