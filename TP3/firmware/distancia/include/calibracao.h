// Constantes da calibração. Gerado à mão ou por:
//   python3 scripts/analise.py dados/ --header firmware/distancia/include/calibracao.h
// Os valores abaixo são PROVISÓRIOS: substituir pelos da aba Ajuste da
// planilha (ou pelos que o analise.py escrever).
#pragma once

// Modelo usado na conversão: 1 = roteiro, 2 = lei de potência, 3 = tabela.
#define MODELO 1

// Tempos de acionamento (us) e conversões por leitura.
const unsigned long T_ON_US = 1000000UL;
const unsigned long T_OFF_US = 1000000UL;
const int M_CONV = 8;

// M1: d = G0 + G1 / sqrt(S)
const float G0 = 0.0;    // mm
const float G1 = 380.0;  // mm * contagem^0.5

// M2: S = ALFA * d^(-N_EXP)  =>  d = (ALFA / S)^(1 / N_EXP)
const float ALFA = 140000.0;
const float N_EXP = 2.0;

// M3: pares (S, d) da faixa de ajuste, com S decrescente.
const int TAB_N = 3;
const float TAB_S[TAB_N] = {900.0, 100.0, 10.0};
const float TAB_D[TAB_N] = {12.0, 38.0, 120.0};

// Limites da faixa de validade.
const float S_LIM = 9.0;    // abaixo disso o sinal não se separa do fundo
const float D_DET = 120.0;  // mm, alcance de detecção: impresso quando S < S_LIM
const float D_MIN = 12.0;   // mm, início da faixa de ajuste: impresso na saturação
const int ADC_SAT = 60;     // adc_on <= isto: fototransistor saturado (-1 = nunca)
