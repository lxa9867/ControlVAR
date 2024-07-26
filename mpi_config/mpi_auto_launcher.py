import subprocess
import os
import time

def execute():
    while True:
        output = subprocess.check_output('kubectl apply -f mpi.yaml', shell=True, stderr=subprocess.STDOUT)
        print(output)
        print('##################')
        time.sleep(100)
        output = subprocess.check_output('kubectl get pods', shell=True, stderr=subprocess.STDOUT).decode('utf-8')
        print(output)
        print('##################')
        if output.count("Error") + output.count("Pending") >= 1:
            output = subprocess.check_output('kubectl delete mpijob mpi', shell=True, stderr=subprocess.STDOUT)
            print(output)
            print('################## relaunch')
        else:
            print(output)
            print('################## finish')
            return


if __name__ == '__main__':
    execute()