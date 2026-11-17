#!/usr/bin/env python3
"""Coleta de ensaios do TP4: manda comandos ao firmware e salva o CSV.

Roda no computador onde a Mega esta ligada (feche o monitor serial antes: dois
programas nao dividem a porta). Abrir a porta reinicia a placa; o script espera
o boot antes de mandar os comandos.

Saida, em dados/ (ou --saida):
  <nome>.csv          linhas de CSV do firmware (cabecalho + amostras)
  <nome>_tabela.csv   linhas '# TAB,...' (comando N, busca do neutro), se houver
  <nome>.meta.txt     data, comandos enviados e comentarios ('#') do firmware,
                      inclusive os parametros (D, N, k, neutro, ganhos, filtro)

Exemplos (da pasta TP4):
  # curva em malha aberta (termina sozinha em '# fim varredura')
  python3 scripts/coleta.py malha_aberta_A50_3 "A 50 3"
  # busca do neutro
  python3 scripts/coleta.py neutro_N1300_1700 "N 1300 1700"
  # degrau em malha aberta para os filtros: 4 s e depois 'x'
  python3 scripts/coleta.py filtros_degrau_o1700 "o 1700" --duracao 4
  # tentativa de sintonia: ganhos e filtro antes, degrau de 150 mm/s por 4 s
  python3 scripts/coleta.py pid_t01_kp2 --pre "f 0" --pre "g 2 0 0" "s 150" --duracao 4
  # ensaios 1, 2 e 3 (terminam sozinhos)
  python3 scripts/coleta.py ensaio1_patamares --pre "q 100 130 160 190 220" "1"
  python3 scripts/coleta.py ensaio2_inversao --pre "r 120" "2"
  python3 scripts/coleta.py ensaio3_distancia --pre "r 120" "3 3"

Ctrl+C durante a coleta manda 'x' (para o motor) e salva o que chegou.
"""
import argparse
import glob
import os
import sys
import time
from datetime import datetime

import serial

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAIS = ("# fim", "# parado", "# E3 estourou", "# nenhuma faixa")
CAMPOS_CSV = 10


def acha_porta():
    portas = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if not portas:
        sys.exit("Nenhuma porta serial encontrada (/dev/ttyUSB* ou /dev/ttyACM*). "
                 "A Mega esta ligada?")
    if len(portas) > 1:
        print(f"Varias portas: {portas}; usando {portas[0]} (use --porta para escolher)")
    return portas[0]


def le_linha(ser):
    bruto = ser.readline()
    if not bruto:
        return None
    return bruto.decode("utf-8", errors="replace").strip()


def envia(ser, cmd, meta, espera=0.3):
    ser.write((cmd + "\n").encode())
    meta.append(f"> {cmd}")
    fim = time.time() + espera
    while time.time() < fim:
        linha = le_linha(ser)
        if linha and linha.startswith("#"):
            meta.append(linha)


def coleta(args):
    saida = os.path.abspath(args.saida)
    os.makedirs(saida, exist_ok=True)
    f_csv = os.path.join(saida, args.nome + ".csv")
    f_tab = os.path.join(saida, args.nome + "_tabela.csv")
    f_meta = os.path.join(saida, args.nome + ".meta.txt")
    if not args.sobrescrever:
        for f in (f_csv, f_meta):
            if os.path.exists(f):
                sys.exit(f"{f} ja existe. Use outro nome ou --sobrescrever.")

    comando = args.comando.strip()
    if comando[0] in "os" and args.duracao is None:
        sys.exit(f"O comando '{comando}' nao termina sozinho: informe --duracao (s).")

    porta = args.porta or acha_porta()
    ser = serial.Serial(porta, 115200, timeout=0.2)
    meta = [f"# data: {datetime.now().isoformat(timespec='seconds')}",
            f"# porta: {porta}"]
    print(f"Porta {porta}; esperando o boot ({args.espera_boot:.1f} s)...")
    time.sleep(args.espera_boot)
    ser.reset_input_buffer()

    for pre in args.pre:
        envia(ser, pre, meta)
    envia(ser, "?", meta)            # registra os parametros no meta

    linhas_csv, linhas_tab = [], []
    ser.reset_input_buffer()
    ser.write((comando + "\n").encode())
    meta.append(f"> {comando}")
    print(f"Coletando '{comando}'... (Ctrl+C para parar)")

    t0 = time.time()
    enviou_x = False
    terminou = False
    try:
        while not terminou:
            agora = time.time() - t0
            if args.duracao is not None and agora >= args.duracao and not enviou_x:
                ser.write(b"x\n")
                meta.append("> x")
                enviou_x = True
            if agora > args.timeout:
                print("Timeout: parando.")
                ser.write(b"x\n")
                meta.append("> x (timeout)")
                break
            linha = le_linha(ser)
            if not linha:
                continue
            if linha.startswith("# TAB,"):
                linhas_tab.append(linha[len("# TAB,"):])
                print(linha)
            elif linha.startswith("#"):
                meta.append(linha)
                print(linha)
                if linha.startswith(FINAIS):
                    terminou = True
            elif linha.startswith("t_s,"):
                if not linhas_csv:
                    linhas_csv.append(linha)
            elif len(linha.split(",")) == CAMPOS_CSV and linha[0].isdigit():
                linhas_csv.append(linha)
                if len(linhas_csv) % 100 == 0:
                    print(f"  {len(linhas_csv) - 1} amostras, t = {linha.split(',')[0]} s")
    except KeyboardInterrupt:
        print("\nInterrompido: mandando 'x'.")
        ser.write(b"x\n")
        meta.append("> x (Ctrl+C)")
    finally:
        ser.write(b"x\n")
        ser.close()

    if linhas_csv:
        with open(f_csv, "w") as f:
            f.write("\n".join(linhas_csv) + "\n")
        print(f"CSV: {f_csv} ({len(linhas_csv) - 1} amostras)")
    if linhas_tab:
        with open(f_tab, "w") as f:
            f.write("\n".join(linhas_tab) + "\n")
        print(f"Tabela: {f_tab} ({len(linhas_tab)} linhas)")
    if not linhas_csv and not linhas_tab:
        print("Nenhum dado recebido. Confira a porta, o firmware gravado e o comando.")
    with open(f_meta, "w") as f:
        f.write("\n".join(meta) + "\n")
    print(f"Meta: {f_meta}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("nome", help="nome base dos arquivos de saida, sem extensao")
    p.add_argument("comando", help='comando do firmware, entre aspas (ex.: "A 50 3")')
    p.add_argument("--pre", action="append", default=[], metavar="CMD",
                   help="comando enviado antes (pode repetir), ex.: --pre 'g 2 0 0'")
    p.add_argument("--duracao", type=float, default=None,
                   help="segundos ate mandar 'x' (obrigatorio para 'o' e 's')")
    p.add_argument("--porta", default=None, help="porta serial (padrao: acha sozinha)")
    p.add_argument("--saida", default=os.path.join(RAIZ, "dados"),
                   help="pasta de saida (padrao: TP4/dados)")
    p.add_argument("--espera-boot", type=float, default=2.5,
                   help="segundos de espera apos abrir a porta (reset da placa)")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="limite total de coleta, em segundos")
    p.add_argument("--sobrescrever", action="store_true")
    coleta(p.parse_args())


if __name__ == "__main__":
    main()
