import tkinter as tk
import plotting_part as pl
import os

# functions
def nol(word= 'input the number of layers you want to construct\n'):
    try:
        tmp = int(input(word))
    except:
        tmp = nol("it's not correct input, please put integer\n")
    return tmp

def see_lights():
    for i in list_of_lights:
        print(i[0:-4])

def see_materials():
    for i in list_of_materials:
        print(i[0:-4])

def start():
    try:
        tmp_layers = []
        for i, j in box:
            tmp = (i.get(), float(j.get()))
            if not tmp[0]+'.csv' in list_of_materials:
                raise ValueError
            tmp_layers.append(tmp)

        light_name, light_theta, light_p = l_name_input.get(), float(l_theta_input.get()), float(l_p_input.get())
        if light_theta < 0 or light_theta > 90 or light_p < 0 or light_p > 1 or not light_name + '.csv' in list_of_lights:
            raise ValueError

        plot.go(light_name, light_theta, light_p, 1-light_p, tmp_layers)

    except ValueError:
        print('Error, please put correct info')

# constants
number_of_layers = nol()

# containers
plot = pl.Core(number_of_layers)
box = []

location = os.getcwd()
for root, dirs, files in os.walk(location + '/lights_i'):
    list_of_lights = files

for root, dirs, files in os.walk(location + '/materials_i'):
    list_of_materials = files

# GUI
window = tk.Tk()
window.title('Multi_Layers')
light_info = tk.Frame(window)
light_info.grid(row= 0, column= 0)

l_name = tk.Frame(light_info)
l_name.grid(row= 0, column= 0)
l_name_Label = tk.Label(l_name, text= 'Select Light').grid(row= 0, column= 0)
l_name_input = tk.Entry(l_name, width = 15)
l_name_input.grid(row= 1, column= 0)

l_theta = tk.Frame(light_info)
l_theta.grid(row= 0, column= 1)
l_theta_Label = tk.Label(l_theta, text= 'Theta(degree)<=90').grid(row= 0, column= 0)
l_theta_input = tk.Entry(l_theta, width = 15)
l_theta_input.grid(row= 1, column= 0)

l_p = tk.Frame(light_info)
l_p.grid(row= 0, column= 2)
l_p_Label = tk.Label(l_p, text= 'Ratio of p-pol(0~1)').grid(row= 0, column= 0)
l_p_input = tk.Entry(l_p)
l_p_input.grid(row= 1, column= 0)

for i in range(1, number_of_layers+1):

    row = tk.Frame(window)
    row.grid(row= i, column= 0, sticky= 'W')
    tk.Label(row, text= 'Layer' + str(i)).grid(row= 0, column= 0, sticky= 'W')

    tk.Label(row, text= 'Name of Material').grid(row= 1, column= 0, sticky= 'W')
    tk.Label(row, text= 'Thickness(nm)').grid(row= 1, column= 2, sticky= 'W')
    material_name = tk.Entry(row, width= 20, bg= 'light green')
    material_name.grid(row= 2, column= 0, sticky= 'W')
    white = tk.Label(row, text= ' '*30).grid(row= 2, column= 1, sticky= 'W')
    thickness = tk.Entry(row, width= 18, bg= 'light green')
    thickness.grid(row= 2, column= 2, sticky= 'W')
    box.append((material_name, thickness))  # Material_name & Thickness objects --> box

tk.Button(row, text= 'Materials', width= 10, command= see_materials).grid(row= 3, column= 0, sticky = 'W')
tk.Button(row, text= 'Lights', width= 10, command= see_lights).grid(row= 3, column= 1, sticky = 'W')
tk.Button(row, text= 'Start', width= 10, command= start).grid(row= 3, column= 2, sticky= 'E')

window.mainloop()
