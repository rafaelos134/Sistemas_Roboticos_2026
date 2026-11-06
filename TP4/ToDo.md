Falta fazer

  1. Registrar o método de medida de D (paquímetro ou volta no papel). O enunciado exige.
  2. Fotos: transmissão engrenada, roda e ligação na Mega com os fios visíveis.
  3. Firmware de controle com:
     - PWM do servo;
     - leitura de v = Δs/Δt usando o Δt medido;
     - filtro;
     - PID com integral limitada e zerada quando o setpoint muda;
     - correção limitada a ±250 µs e pulso entre 1000 e 2000 µs;
     - log em CSV (t, ticks, s, v bruta, v filtrada, setpoint, pulso);
     - comandos seriais para disparar cada ensaio.
  4. Neutro do servo: busca em passos de 10 µs em torno de 1500 µs. Escrever 1500 sem essa busca zera o item.
  5. Curva pulso × velocidade em malha aberta, de 1000 a 2000 µs, nos dois sentidos, em mm/s. Comentar banda morta, saturação e assimetria.
  6. Dois filtros no mesmo degrau (por exemplo média móvel e passa-baixa), com os parâmetros declarados e um gráfico junto da velocidade bruta. Escolher um e justificar.
  7. Sintonia do PID: tabela de tentativas até os ganhos finais, com unidades em µs/(mm/s).
  8. Ensaios de malha fechada:
     - cinco patamares crescentes, cerca de 3 s cada;
     - inversão de sentido;
     - distância s(t) com o valor final, comparada com πD × número de voltas.