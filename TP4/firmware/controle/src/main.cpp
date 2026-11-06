// Teste de sanidade do servo + encoder (Arduino Mega 2560).
//
// Sem comandos: ao ligar, repete para cada pino de sinal (9 e 10):
//   2000 us por 3 s, parado 1500 us por 2 s, 1000 us por 3 s, parado por 2 s
// e imprime, a cada 0,5 s, o pino, o pulso e os ticks do encoder.
// Se o servo girar, os ticks mudam e o pino certo aparece na saida.
//
// Alimente o servo bem (5 V externo se a placa reiniciar) e deixe a roda livre.

#include <Arduino.h>
#include <Encoder.h>
#include <Servo.h>

Encoder enc(2, 3);
Servo servo;

static void fase(uint8_t pino, int us, unsigned long ms) {
  servo.writeMicroseconds(us);
  long ini = enc.read();
  unsigned long t0 = millis();
  while (millis() - t0 < ms) {
    delay(500);
    Serial.print(F("pino=")); Serial.print(pino);
    Serial.print(F("  pulso=")); Serial.print(us);
    Serial.print(F("  ticks=")); Serial.println(enc.read());
  }
  long delta = enc.read() - ini;
  Serial.print(F("  >> variacao na fase: ")); Serial.print(delta);
  Serial.println(abs(delta) > 5 ? F(" ticks  GIROU") : F(" ticks  PARADO"));
}

void setup() {
  Serial.begin(115200);
  Serial.println(F("# teste de sanidade do servo"));
}

void loop() {
  const uint8_t pinos[1] = { 10}; //o pino é o 10
  for (uint8_t i = 0; i < 1; i++) {
    uint8_t p = pinos[i];
    Serial.print(F("# sinal do servo no pino ")); Serial.println(p);
    servo.attach(p, 1000, 2000);
    fase(p, 2000, 3000);
    fase(p, 1500, 2000);
    fase(p, 1000, 3000);
    fase(p, 1500, 2000);
    servo.detach();
  }
  Serial.println(F("# fim do ciclo, repetindo"));
}
