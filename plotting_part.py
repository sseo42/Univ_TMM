import numpy as np
import matplotlib.pyplot as plt
import graph_to_color as gtc


class Core:
    def __init__(self, num):
        self.num = num  # number of layers
        self.wave_vector_const = np.array([20*np.pi/i for i in range(3900,8301)])  # for k_z-wave_vector, (nm), from 390nm to 830nm per 0.1nm
        self.init_layer =[1 for i in range(4401)]  # size of wave vectors
        self.zero_layer = [0 for i in range(4401)]
        self.init_theta = None  # for self-indexing

        self.medium_last_layer = True
        self.sum_num = 100
        self.gtc = gtc.ColorFinder(self.sum_num)  # wave length to color class

    def clear(self):
        print('not yet')

    def go(self, light, theta_org, p, q, layers):
        theta = theta_org*np.pi/180  # degree to radian
        d_d, s_s, p_p = [], [], []  # in materials-(p,s), on materials-s, on materials-p
        b_layer, alpha = self.init_layer.copy(), [theta for i in range(4401)]  # initial refractive index, initial theta for every wavelength

        for name, thickness in layers:
            wave_length, refractive_i, extinction_c = [], [], []
            with open('materials_i/' + name + '.csv') as f:
                while True:
                    try:
                        a, b, c = map(float, f.readline().split(','))
                        wave_length.append(a)
                        refractive_i.append(b)
                        extinction_c.append(c)

                    except ValueError:
                        break

            ref = self.binary_search(wave_length, refractive_i, True, extinction_c)  # (4401)numpy-array of complex_refractive index

            b_layer, alpha, dd, ss, pp = self.layer_constructor(b_layer, alpha, ref, thickness)  # current ref, current theta, delta, s-polo, p-polo/(last-current)
            d_d.append(dd)
            s_s.append(ss)
            p_p.append(pp)

        last_ref, last_theta, dump2, last_s, last_p = self.layer_constructor(b_layer, alpha, self.init_layer, 0)  # for last material-air
        s_s.append(last_s)
        p_p.append(last_p)
        # d_d = (layer num, 2, 2, 4401) s_s,p_p = (layer num + 1, 2, 2, 4401)

        n_in_total_E, n_out_total_E, n_t_s_ans, n_t_p_ans, n_r_s_ans, n_r_p_ans = self.forward(light, np.array([theta]*4401), p, q, d_d, s_s, p_p, last_theta, last_ref)  # 4401 array

        if self.medium_last_layer == True:
            in_total_E, out_total_E, t_s_ans, t_p_ans, r_s_ans, r_p_ans = [0]*(4401-self.sum_num), [0]*(4401-self.sum_num), [0]*(4401-self.sum_num), \
                                                                          [0]*(4401-self.sum_num), [0]*(4401-self.sum_num), [0]*(4401-self.sum_num)

            for i in range(self.sum_num):
                t_s_ans += n_t_s_ans[i:i+4401-self.sum_num]
                t_p_ans += n_t_p_ans[i:i+4401-self.sum_num]
                r_s_ans += n_r_s_ans[i:i+4401-self.sum_num]
                r_p_ans += n_r_p_ans[i:i+4401-self.sum_num]
                in_total_E += n_in_total_E[i:i+4401-self.sum_num]
                out_total_E += n_out_total_E[i:i+4401-self.sum_num]

            t_s_ans /= self.sum_num
            t_p_ans /= self.sum_num
            r_s_ans /= self.sum_num
            r_p_ans /= self.sum_num
            in_total_E /= self.sum_num
            out_total_E /= self.sum_num

            x = [i/10 for i in range(3900 + int(self.sum_num/2), 8301 - int(self.sum_num/2))]

        else:
            t_s_ans = n_t_s_ans
            t_p_ans = n_t_p_ans
            r_s_ans = n_r_s_ans
            r_p_ans = n_r_p_ans
            in_total_E = n_in_total_E
            out_total_E = n_out_total_E

            x = [i / 10 for i in range(3900, 8301)]



        in_color, in_x, in_y = self.gtc.rgb(in_total_E)
        out_color, out_x, out_y = self.gtc.rgb(out_total_E)

        '''
        plt.title('P-pol & S-pol')
        plt.plot(x, t_s_ans, label='S-Transmittance', linestyle= '--', color='Blue')
        plt.plot(x, r_s_ans, label='S-Relectance', color='Blue')
        plt.plot(x, t_p_ans, label = 'P-Transmittance', linestyle='--', color = 'Red')
        plt.plot(x, r_p_ans, label='P-Relectance', color='Red')
        plt.legend(loc='best')
        plt.show()
        '''


        '''
        flg = plt.figure('Incident_angle:' + str(theta_org) + '    P-pol:' + str(p * 100) + '%' + '  S-pol:' + str(q * 100) + '%', tight_layout=True)

        x1 = flg.add_subplot(2, 2, 1)
        x2 = flg.add_subplot(2, 2, 2)
        x3 = flg.add_subplot(2, 2, 3)
        x4 = flg.add_subplot(2, 2, 4)

        x1.set_xlabel('WaveLength(nm)')
        x1.set_ylabel('Spectral Power Distribution')
        x1.plot(x, out_total_E, label = 'out_colored', color = out_color)
        x1.legend(loc='best')

        x2.set_xlabel('WaveLength(nm)')
        x2.set_ylabel('Spectral Power Distribution')
        x2.plot(x, in_total_E, label = 'in_colored', color = in_color)
        x2.legend(loc='best')

        x3.set_xlabel('WaveLength(nm)')
        x3.set_ylabel('P-pol')
        x3.plot(x, t_p_ans, label = 'Transmittance', color = 'red')
        x3.plot(x, r_p_ans, label='Relectance', linestyle='--', color='blue')
        x3.legend(loc='best')

        x4.set_xlabel('WaveLength(nm)')
        x4.set_ylabel('S-pol')
        x4.plot(x, t_s_ans, label = 'Transmittance', color = 'red')
        x4.plot(x, r_s_ans, label = 'Relectance', linestyle= '--', color = 'blue')
        x4.legend(loc='best')

        plt.show()
        '''


        plt.scatter(in_x, in_y)
        plt.xlim(-0.1,0.8)
        plt.ylim(-0.1, 0.9)

        plt.show()

        '''
        plt.plot(x, t_p_ans, color = 'red')
        plt.plot(x, r_p_ans, color = 'blue')
        plt.xlabel('WaveLength(nm)')
        plt.ylabel('P-pol')
        plt.show()'''

    def layer_constructor(self, b_layer, theta, ref, thickness):  # last refractive index, last theta, current refractive index, current thickness(nm)

        theta_2 = np.arcsin(b_layer*np.sin(theta)/ref) #check

        cos_theta = np.cos(theta)
        cos_theta_2 = np.cos(theta_2)

        one_to_one = b_layer*cos_theta
        one_to_two = b_layer*cos_theta_2
        two_to_one = ref*cos_theta
        two_to_two = ref*cos_theta_2

        delta1 = np.exp(-1j * self.wave_vector_const * two_to_two * thickness)  # 4401-array
        delta2 = np.exp(1j * self.wave_vector_const * two_to_two * thickness)
        rs = (one_to_one - two_to_two)/(one_to_one + two_to_two)
        rp = (two_to_one - one_to_two)/(two_to_one + one_to_two)
        ts = 2*one_to_one/(one_to_one + two_to_two)
        tp = 2*one_to_one/(one_to_two + two_to_one)

        tmp_s, tmp_p = rs/ts, rp/tp

        dd = [[delta1, self.zero_layer], [self.zero_layer, delta2]]
        ss = [[self.init_layer/ts, tmp_s], [tmp_s, self.init_layer/ts]]  # 3dim-array(2X2X4401)
        pp = [[self.init_layer/tp, tmp_p], [tmp_p, self.init_layer/tp]]

        return ref, theta_2, dd, ss, pp

    def forward(self, light, theta, p, q, d_d, s_s, p_p, last_theta, last_ref):  # name, 4401-array, init p polo, init s polo, delta, s polo, p polo, last theta 4401 array
        tmp_w, tmp_e, alpha = [], [], theta

        with open('lights_i/' + light + '.csv') as f:
            while True:
                try:
                    a, b = map(float, f.readline().split(','))
                    tmp_w.append(a)
                    tmp_e.append(b)

                except ValueError:
                    break

        light_e = self.binary_search(tmp_w, tmp_e, False, False)  # 4401-array

        p_light_e, s_light_e = np.array([p]*4401)*light_e, np.array([q]*4401)*light_e

        d_d = np.transpose(d_d, (3, 0, 1, 2))
        s_s = np.transpose(s_s, (3, 0, 1, 2))  # (layer num + 1, 2, 2, 4401) to (4401, layer num + 1, 2, 2)
        p_p = np.transpose(p_p, (3, 0, 1, 2))

        ts, tp, rs, rp = [], [], [], []
        for i in range(4401):
            m_s = s_s[i][0]
            m_p = p_p[i][0]
            for ii in range(self.num):
                m_s = np.dot(np.dot(m_s, d_d[i][ii]), s_s[i][ii+1])
                m_p = np.dot(np.dot(m_p, d_d[i][ii]), p_p[i][ii+1])


            ts.append(1 / m_s[0][0])
            tp.append(1 / m_p[0][0])
            rs.append(m_s[1][0] / m_s[0][0])
            rp.append(m_p[1][0] / m_p[0][0])


        mts, mtp, mrs, mrp = np.mat(ts), np.mat(tp), np.mat(rs), np.mat(rp)
        init_ref = np.array(self.init_layer)

        t_s_sq, t_p_sq = np.array(ts)*np.array(mts.H).flatten()*np.real(last_ref*np.cos(last_theta))/np.real(init_ref*np.cos(theta)), \
                         np.array(tp)*np.array(mtp.H).flatten()*np.real(last_ref*np.cos(1j*last_theta))/np.real(init_ref*np.cos(1j*theta))
        r_s_sq, r_p_sq = np.array(rs)*np.array(mrs.H).flatten(), np.array(rp)*np.array(mrp.H).flatten()

        in_total_E = s_light_e*r_s_sq*np.real(init_ref*np.cos(theta)) + p_light_e*r_p_sq*np.real(init_ref*np.cos(1j*theta))
        out_total_E = p_light_e*t_p_sq*np.real(init_ref*np.cos(1j*theta)) + s_light_e*t_s_sq*np.real(init_ref*np.cos(theta))

        return in_total_E, out_total_E, t_s_sq, t_p_sq, r_s_sq, r_p_sq

    def binary_search(self, wave_length, refractive_i, is_m, extinction_c):
        last_idx = len(wave_length) - 1
        vals = []

        if not is_m:  # when there's no extinction_c
            for i in range(3900, 8301):
                tmp = i / 10
                l, r = self.b_s_l(tmp, 0, last_idx, wave_length), self.b_s_r(tmp, 0, last_idx, wave_length)

                if l == r:
                    ans = refractive_i[l]
                elif l < r:
                    ttmp = (tmp - wave_length[l]) / (wave_length[r] - wave_length[l])
                    a, b = refractive_i[l], refractive_i[r]
                    ans = (b - a)*ttmp + a
                else:
                    raise ValueError

                vals.append(ans)
            return vals

        else:
            for i in range(3900, 8301):
                tmp = i / 10
                l, r = self.b_s_l(tmp, 0, last_idx, wave_length), self.b_s_r(tmp, 0, last_idx, wave_length)

                if l == r:
                    ans = complex(refractive_i[l], extinction_c[l])
                elif l < r:
                    ttmp = (tmp - wave_length[l])/(wave_length[r]-wave_length[l])
                    a, b, c, d = refractive_i[l], refractive_i[r], extinction_c[l], extinction_c[r]
                    ans = complex((b - a)*ttmp + a, (d-c)*ttmp + c)
                else:
                    raise ValueError

                vals.append(ans)
        return vals

    def b_s_r(self, target, a, b, wave_length):
        if a == b:
            return b

        mid = (a + b)//2
        if target <= wave_length[mid]:
            return self.b_s_r(target, a, mid, wave_length)
        else:
            return self.b_s_r(target, mid + 1, b, wave_length)

    def b_s_l(self, target, a, b, wave_length):
        if a == b:
            return a

        mid = (a + b)//2 + 1
        if target >= wave_length[mid]:
            return self.b_s_l(target, mid, b, wave_length)
        else:
            return self.b_s_l(target, a, mid - 1, wave_length)
