# rucna proba, vrijeme + determinizam 
import time
from src.pipeline import load_config, run

config = load_config('configs/baseline.yaml')
t0 = time.time()
result = run(config)
elapsed = time.time() - t0

print('naj duzina ture:', result['best_tour_length'])
print('poznat opt za berlin52: 7542')
print('best params:', result['best_params'])
print(f'ttc: {elapsed:.1f}s')

"""
uncomment za determinizam test


from src.pipeline import load_config, run

config = load_config('configs/smoke_test.yaml')
r1 = run(config)
r2 = run(config)
assert r1['best_tour_length'] == r2['best_tour_length'], 'FALSE NONDETERM'
print('OK - determizam pass')
"""

