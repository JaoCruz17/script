# Radar meteorológico e ladder Polymarket — SBGR

O script mantém o painel de terminal existente e calcula `P(YES)` pela frequência dos membros do ensemble, em vez de ajustar uma normal quando os dados dos membros estão disponíveis. ECMWF, GFS e ICON são combinados somente se a validação histórica em SBGR aprovar a mistura.

## Instalação e início (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python temperatura_sbgr.py --date 2026-10-05
```

Para atualizar continuamente, use `--watch --interval 300`. O intervalo mínimo aceito é 60 segundos. Para encerrar, pressione `Ctrl+C`.

## Como a combinação funciona

- Busca os ensembles atuais ECMWF IFS, GFS GEFS e DWD ICON EPS para as coordenadas de SBGR.
- Baixa até 180 dias de máximas METAR arquivadas de SBGR e previsões anteriores Open-Meteo para o mesmo horizonte (D-0 a D-7).
- Corrige o viés de cada modelo e procura pesos não negativos, limitados entre 10% e 80%, usando apenas dados anteriores em cada passo do walk-forward.
- Ativa a mistura apenas com pelo menos 30 dias pareados e se o RMSE walk-forward da mistura ficar pelo menos 2% abaixo do ECMWF corrigido.
- Se o teste falhar, o horizonte não estiver disponível ou algum ensemble atual faltar, `P(YES)` e EV continuam usando somente os membros do ECMWF, sem aplicar pesos não validados.
- Quando a mistura é aprovada, junta os membros dos três ensembles com os pesos obtidos e aplica as correções de viés antes de calcular probabilidades e EV.
- A recomendação pode ser uma faixa única ou duas faixas adjacentes, e só aparece quando o EV líquido estimado após a taxa taker for positivo. Não executa ordens.

## Limitações

As métricas de validação comparam máximas horárias de previsões anteriores do ensemble-mean com as máximas METAR arquivadas. A distribuição ao vivo, por sua vez, usa os membros de máximas diárias; portanto, o backtest é um teste do desempenho dos modelos por família e horizonte, não uma validação histórica completa de cada membro ou das probabilidades da ladder. Os arquivos públicos também podem não ter cobertura para todos os dias. Pesos aprovados não garantem lucro, e o EV exibido exclui slippage.

Os dados vêm das APIs públicas do [Open-Meteo](https://open-meteo.com/) e do arquivo METAR ASOS da [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/). A disponibilidade dessas fontes é necessária para calibrar; se estiverem indisponíveis, o painel informa o motivo e permanece no ECMWF.

## Testes

```powershell
python -m unittest -v test_model_engine.py
```
