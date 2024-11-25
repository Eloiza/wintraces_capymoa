import numpy as np
np.float = np
np.int = np

import matplotlib.pyplot as plt 
from capymoa.drift.detectors import ADWIN, DDM
from capymoa.drift.detectors import ADWIN as capy_adwin
from skmultiflow.drift_detection.adwin import ADWIN as sk_adwin
from river import drift as river_drift


np.random.seed(0)
# detector = DDM()

# ####################################################################
# adwin = sk_adwin()
# albert_drift = []
# all_accs = []
# window = []

# # Simulating a data stream as a normal distribution of 1's and 0's
# data_stream = np.random.randint(2, size=2000)
# # Changing the data concept from index 999 to 2000
# for i in range(999, 2000):
#     data_stream[i] = np.random.randint(4, high=8)

# # Adding stream elements to ADWIN and verifying if drift occurred
# for i in range(2000):
#     old = adwin.estimation    
#     adwin.add_element(data_stream[i])
#     new = adwin.estimation
#     if adwin.detected_change() and old > new:
#         print('Albert Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
#         albert_drift.append(i)

#     window.append(data_stream[i])
#     all_accs.append(sum(window) / len(window))      

#     if len(window) > 32:
#         window.pop(0)


# plt.plot([i for i in range(len(all_accs))], all_accs)    
# for x_point in albert_drift:
#     plt.axvline(x_point, color='red', linestyle='dashed')

# plt.title("Acuracia ao longo do tempo")    
# plt.show()

###################################################################3
data_stream = []
data_stream = np.random.randint(2, size=5000)

detector_sk = sk_adwin()
# detector_river = river_drift.ADWIN()
detector_river = river_drift.ADWIN()
detector_capymoa = capy_adwin()
    
for i in range(1999, 3000):
    data_stream[i] = np.random.randint(4, high=8)
    # data_stream[i] = np.random.randint(4, high=8)

for i in range(2999,4000):
    data_stream[i] = np.random.randint(2)
    
for i in range(3999, 5000):
    data_stream[i] = np.random.randint(10, high=20)
    
        
window = []
all_accs = []
drifts = []
drift_sk = []   
drift_river = []
for i in range(5000):
    detector_capymoa.add_element(data_stream[i])
    if detector_capymoa.detected_change():
        print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
        drifts.append(i)

    old = detector_sk.estimation
    detector_sk.add_element(data_stream[i])
    new = detector_sk.estimation
    if detector_sk.detected_change() and old > new:
        drift_sk.append(i) 
    
    detector_river.update(data_stream[i]) 
    if detector_river.drift_detected:
        drift_river.append(i)        
                
    # if detector_sk.detected_change():
    #     drift_sk.append(i) 

    window.append(data_stream[i])
    all_accs.append(sum(window) / len(window))      

    if len(window) > 32:
        window.pop(0)


plt.plot([i for i in range(len(all_accs))], all_accs)    
# for x_point in drifts:
#     plt.axvline(x_point, color='red', linestyle='dashed')

print("drifts capymoa", drifts)
print("drifts sklearn", drift_sk)
print("drifts river", drift_river)

for x_point in drift_river:
    plt.axvline(x_point, color='black', linestyle='solid')

for x_point in drift_sk:
    plt.axvline(x_point, color='red', linestyle='dashed')

plt.title("Acuracia ao longo do tempo")    
plt.show()