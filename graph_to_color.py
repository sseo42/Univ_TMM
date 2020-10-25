import numpy as np


class ColorFinder:
    def __init__(self, sum_num):

        # constants
        self.c, self.dielectric_c = 3*1e8, 8.854*1e-12

        # wave length & rgb_color_matching
        self.length, xyz = [], []
        t = np.array([[0.41847, -0.15866, -0.082835], [-0.091169, 0.25243, 0.015708], [0.0009209, -0.0025498, 0.17860]])  # xyz to rgb

        with open('lin2012xyz10e_fine_7sf.csv', 'r') as f:
            while True:
                try:
                    l, x, y, z = map(float, f.readline().split(','))
                    self.length.append(l)
                    xyz.append([x,y,z])
                except ValueError:
                    break

        xyz = xyz[0:4401 - sum_num]
        self.xyz = np.array(xyz)

        self.color_matching = np.dot(t, self.xyz.T).T  # wavelength X rgb(4401X3)

    def RGB(self, power):

        return np.dot(power, self.color_matching)

    def rgb(self, power):
        rgb_info = self.RGB(np.real(power))

        if rgb_info[0] < 0 or rgb_info[1] < 0 or rgb_info[2] < 0:
            print("it can't be showed by rgb matching function")

        r, g, b = np.abs(rgb_info) # if there is negative value it means it can't be showed by rgb matching function
        denominator = r + g + b
        tmp1, tmp2, tmp3 = np.dot(np.real(power), self.xyz)
        tmp_sum = tmp1 + tmp2 + tmp3
        print(tmp1/tmp_sum, tmp2/tmp_sum, tmp3/tmp_sum)


        return [r/denominator, g/denominator, b/denominator], tmp1/tmp_sum, tmp2/tmp_sum


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_wavelength = [i / 10 for i in range(3900, 8301)]
    test_ans = ColorFinder().color_matching.T

    plt.title('RGB_converter')
    plt.plot(test_wavelength, test_ans[0], label='r_graph', color='r')
    plt.plot(test_wavelength, test_ans[1], label='g_graph', color='g')
    plt.plot(test_wavelength, test_ans[2], label='b_graph', color='b')
    plt.xlabel('wave_length')
    plt.legend(loc='best')
    plt.show()