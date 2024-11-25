from capymoa.drift.detectors import ADWIN, DDM
import matplotlib.pyplot as plt 

def main():
    detector = EDDM()
    acc = 0 
    all_accs = []
    print("Rodando janela 1")
    drifts = []
    
    window = []
    for i in range(1000):
        # 40-50
        if i % 2:
            value = 1
        else:
            value = 0
            
        detector.add_element(value)
        
        has_drifted = detector.detected_change()
        if has_drifted:
            print(f"\tDrift {i}")
            drifts.append(i)
            detector.reset(True)
            
        acc += value 
        
        window.append(value)
        all_accs.append(sum(window) / len(window))      

        if len(window) > 32:
            window.pop(0)
    
    print(f"Mean win 1: {acc/1000}")    
    print("Rodando janela 2")
    acc = 0 
    # 50-40
    for i in range(1000):
        if i % 2:
            value = 1
        else:
            value = 0
        
        detector.add_element(value)
        
        has_drifted = detector.detected_change()
        if has_drifted:
            print(f"\tDrift {i+1000}")
            drifts.append(i+1001)
            detector.reset(True)

        acc += value 
        window.append(value)
        all_accs.append(sum(window) / len(window))
        if len(window) > 32:
            window.pop(0)



    print(f"Mean win 2: {acc/1001}")
    print("Rodando janela 3")
    acc =0     
    for i in range(1000):
        value = 1
            
        detector.add_element(value)
        
        has_drifted = detector.detected_change()
        if has_drifted:
            print(f"\tDrift {i+2000}")
            drifts.append(i+2001)
            detector.reset(True)

        acc += value 
        window.append(value)
        all_accs.append(sum(window) / len(window))
        if len(window) > 32:
            window.pop(0)
            

    print(f"Mean win 3: {acc/1001}")
    
    print("Rodando janela 4")
    acc = 0
    for i in range(1000):
        value = 0
            
        detector.add_element(value)
        
        has_drifted = detector.detected_change()
        if has_drifted:
            print(f"\tDrift {i+3000}")
            drifts.append(i+3001)
            detector.reset(True)

        acc += value 
        window.append(value)
        all_accs.append(sum(window) / len(window))
        if len(window) > 32:
            window.pop(0)

    print(f"Mean win 4: {acc/1001}")
    
    print("Rodando janela 5")
    acc = 0 
    # 50-40
    for i in range(1000):
        if i % 2:
            value = 1
        else:
            value = 0
            
        detector.add_element(value)
        
        has_drifted = detector.detected_change()
        if has_drifted:
            print(f"\tDrift {i+4000}")
            drifts.append(i+4001)
            detector.reset(True)

        acc += value 
        window.append(value)
        all_accs.append(sum(window) / len(window))
        if len(window) > 32:
            window.pop(0)

    print(f"Mean win 5: {acc/1001}")
    
    plt.plot([i for i in range(len(all_accs))], all_accs)    
    for x_point in drifts:
        plt.axvline(x_point, color='red', linestyle='dashed')

    plt.title("Acuracia ao longo do tempo")    
    plt.show()
    
    # 40-90
if __name__ == "__main__":
    main()
    #90-50

    #50-30
        