import numpy as np

def generate_data(count=1000000):
    a = np.random.uniform(-10.0, 10.0, (count, 16)).astype(np.float32)
    b = np.random.uniform(-10.0, 10.0, (count, 16)).astype(np.float32)

    np.savetxt("m_a.txt", a)
    np.savetxt("m_b.txt", b)

    print(f"generated {count} pairs of matrices 4x4") 

if __name__ == "__main__":
    generate_data()
