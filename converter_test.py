import matplotlib.pyplot as plt
from graph_to_color import color_finder as cvt

test_wavelength = [i/10 for i in range(3900, 8301)]
test_ans = cvt().color_matching.T

plt.title('RGB_converter')
plt.plot(test_wavelength, test_ans[0], label= 'r_graph', color= 'r')
plt.plot(test_wavelength, test_ans[1], label= 'g_graph', color= 'g')
plt.plot(test_wavelength, test_ans[2], label= 'b_graph', color= 'b')
plt.xlabel('wave_length')
plt.legend(loc= 'best')
plt.show()