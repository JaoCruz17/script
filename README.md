# Radar meteorológico e ladder Polymarket — SBGR

Script de terminal para acompanhar METAR de SBGR, previsões ECMWF/GFS/ICON e cotações públicas da Polymarket. Não executa ordens nem usa chaves privadas.

## Instalar e iniciar no PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python temperatura_sbgr.py --date 2026-10-07
```

Atualização contínua a cada cinco minutos:

```powershell
python temperatura_sbgr.py --date 2026-10-07 --watch --interval 300
```

Prior padrão do conjunto: ECMWF=50%, GFS=30%, ICON=20%. Ele é o fallback e a referência conservadora; quando há histórico suficiente, o script tenta aprender pesos por horizonte. Para configurar o prior:

```powershell
python temperatura_sbgr.py --date 2026-10-07 --weights ECMWF=0.4,GFS=0.4,ICON=0.2
```

`--weights` precisa listar os três modelos e somar 1. Os pesos configurados são o prior; eles não representam pesos ótimos comprovados. Se um modelo falhar ao buscar os membros ao vivo, o peso adaptativo/prior é renormalizado entre os modelos disponíveis. Se nenhum deles tiver peso positivo, o script distribui peso igualmente entre os modelos disponíveis. O terminal mostra os pesos do prior, os pesos aplicados e se o ajuste adaptativo foi aceito ou se o sistema voltou ao prior.

O tamanho de ordem usado para estimar slippage é 10 contratos por padrão; altere com `--contracts 25`. O intervalo mínimo do monitoramento é 60 segundos.

## Previsão e comparação dos modelos

- As probabilidades ao vivo são a frequência ponderada dos membros que caem em cada faixa, respeitando o modo de fronteira configurado para o mercado.
- O painel exibe média e desvio de cada modelo, média/desvio do conjunto, P5/P50/P95, intervalo de previsão P2,5–P97,5, número de membros e horário da consulta.
- A pontuação de consenso usa a diferença entre as médias: média dos termos `exp(-diferença/2°C)` para cada par, convertida para 0–100. Os níveis são alta (dispersão máxima ≤1°C), moderada (≤2°C) e baixa (>2°C); pares que divergem pelo menos 1°C aparecem como zonas de desacordo. É uma heurística de concordância, não uma probabilidade de acerto e não altera P(YES) ou EV.
- O intervalo P2,5–P97,5 é um intervalo de previsão dos cenários, não um intervalo de confiança da média. O nível de concordância entre médias é heurístico e não altera probabilidades ou EV.
- Se um modelo falhar, os outros continuam. Se todos falharem, o painel informa indisponibilidade e não calcula probabilidades.

### Pesos adaptativos por antecedência

Para cada horizonte D-0 a D-7, o script tenta escolher pesos numa grade limitada a 10%–90% por modelo, usando erros de previsões arquivadas e apenas dados anteriores ao dia avaliado. A otimização considera RMSE, MAE, recência (meia-vida de 30 dias), correlação entre erros e regularização em direção ao prior configurado. Os pesos novos só são usados se o objetivo histórico walk-forward melhorar pelo menos 1% sobre o prior; senão, mantém-se o prior. Com apenas um modelo ativo, ele recebe 100%. Se faltar histórico, os pesos voltam ao prior e são renormalizados entre os modelos disponíveis.

O ajuste requer ao menos 30 datas para estimar correções e 30 erros fora da amostra. O backtest probabilístico separado requer ao menos 105 datas comuns para reservar 60 datas iniciais, aquecer a distribuição de erros por 15 datas e avaliar pelo menos 30 datas. A janela recente de ajuste é de até 90 datas; resultados são armazenados em cache por seis horas. O horizonte é específico, mas o ajuste não separa estação do ano, regime meteorológico ou faixas de lead menores que um dia. A amostra de 180 dias pode ser pequena; nesse caso, a saída informa o motivo e usa o prior.

## Backtest probabilístico

Para antecedências D-0 a D-7, o script alinha as previsões de máxima horária arquivadas dos modelos com histórico disponível e as máximas diárias observadas no arquivo METAR de SBGR. Usa somente datas anteriores para corrigir viés e gerar uma distribuição empírica walk-forward dos erros. Só exibe métricas quando há pelo menos 105 datas comuns: 60 datas iniciais, 15 dias de aquecimento da distribuição de erros e 30 datas de avaliação.

Compara ECMWF, GFS, ICON e mistura nos mesmos dias e mostra Brier multicategoria em bins inteiros `[n,n+1)°C`, log loss com suavização de `1e-6`, CRPS em °C e cobertura observada do intervalo P5–P95. Esse backtest é uma validação probabilística derivada de erros históricos dos modelos, **não** uma validação com membros de ensembles históricos. A avaliação por bins inteiros também é genérica; só representa o contrato se suas regras usarem exatamente essas fronteiras. Se houver dados insuficientes, a limitação aparece no painel.

## Regras de resolução — necessárias para P(YES) e EV

Antes de atribuir probabilidades/EV a uma faixa, confira no contrato a fonte de resolução, estação, fuso, unidade, arredondamento/precisão e período diário. O script não presume que “São Paulo” significa SBGR.

O arquivo [`market_resolution.json`](market_resolution.json) começa deliberadamente com `confirmed: false` e campos em branco. Assim, o painel pode mostrar previsão e livro, mas mantém P(YES), EV e recomendação suspensos até a configuração ser preenchida após a leitura das regras. Registre a fonte e URL oficiais, estação explicitamente citada, fuso, unidade `C`, precisão, período local e uma das regras de fronteira suportadas:

- `interval_start`: uma faixa `24°C` significa `[24,25)`; “24°C or below” termina em 25°C e “24°C or above” começa em 24°C.
- `nearest_degree`: `24°C` significa `[23.5,24.5)`; as faixas de cauda e ranges seguem a mesma convenção.

Para a implementação atual, `precision` precisa ser `1°C`. Marque `confirmed` como `true` somente depois de conferir o contrato. O programa também exige que o texto obtido da API mencione a estação configurada e que as faixas da ladder sejam interpretáveis, contíguas e sem sobreposição. Se a regra usar outra unidade, estação, precisão ou semântica, não force a configuração: o cálculo permanece bloqueado até o código suportá-la.

## Livro, taxas e qualidade das cotações

- Busca books YES e NO no CLOB e estima o preço médio para o número configurado de contratos, caminhando pelos níveis de ask.
- Slippage é a diferença entre o ask mais baixo e o preço médio estimado. EV usa preço médio, taxa taker calculada nível a nível e probabilidade modelada; EV% divide o EV líquido pelo custo total.
- Se o book não cobrir todo o tamanho pedido ou a taxa for desconhecida, o EV desse lado fica em branco. Books com timestamp acima de 15 minutos são marcados como antigos e excluídos do EV. Quando a API não fornece timestamp, o estado mostra `idade ?` para deixar a limitação visível.
- A coluna de preço implícito usa o meio do spread quando há bid e ask; é uma referência, não o custo executável.
- A ladder é verificada para caudas, lacunas e sobreposições. Com regras confirmadas, a soma das probabilidades das faixas deve ficar próxima de 100%; caso contrário, o painel mostra um alerta.

Taxa e slippage observados não cobrem mudanças futuras no livro, impacto de mercado além da profundidade consultada nem risco de resolução. EV positivo não garante lucro.

## Fontes e limitações

Previsões atuais: [Open-Meteo Ensemble API](https://open-meteo.com/en/docs/ensemble-api). Histórico de previsões: [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api). Observações históricas: arquivo METAR ASOS da [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/). Mercado e books: APIs públicas Gamma e CLOB da Polymarket.

O histórico de modelos pode ter cobertura diferente por família e antecedência. Métricas calculadas no arquivo SBGR são evidência histórica, não garantia de desempenho futuro nem de correspondência com a resolução do mercado.
