import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from tkinter import simpledialog
from tkinter import messagebox

class NuclearExplosionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Симуляция Ядерного Взрыва и Бифуркационная Диаграмма")

        # Создаем вкладки
        self.tab_control = ttk.Notebook(master)

        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)

        self.tab_control.add(self.tab1, text='График')
        self.tab_control.add(self.tab2, text='Бифуркационная диаграмма')

        self.tab_control.pack(expand=1, fill="both")

        # Элементы управления и фигура для вкладки График
        self.setup_tab1()

        # Элементы управления и фигура для вкладки Бифуркационная диаграмма
        self.setup_tab2()

    def setup_tab1(self):
        # Добавляем элементы управления на tab1
        self.setup_controls(self.tab1)

        # Создаем место для графика в tab1
        fig, self.ax1 = plt.subplots()
        self.canvas1 = FigureCanvasTkAgg(fig, self.tab1)
        self.canvas1.get_tk_widget().grid(row=9, column=0, columnspan=3)

    def setup_tab2(self):
        # Метка для выбора параметра
        self.parameter_label = ttk.Label(self.tab2, text="Параметр:")
        self.parameter_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        # Выпадающий список для выбора параметра
        self.parameter_combobox = ttk.Combobox(self.tab2, values=["A", "P", "e", "R"])
        self.parameter_combobox.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        # Метка для выбора пластины
        self.plate_label = ttk.Label(self.tab2, text="Пластина:")
        self.plate_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

        # Выпадающий список для выбора пластины
        self.plate_combobox = ttk.Combobox(self.tab2, values=["Верхняя", "Нижняя"])
        self.plate_combobox.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        # Ввод минимального значения параметра
        self.min_label = ttk.Label(self.tab2, text="Min:")
        self.min_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.min_entry = ttk.Entry(self.tab2)
        self.min_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        # Ввод максимального значения параметра
        self.max_label = ttk.Label(self.tab2, text="Max:")
        self.max_label.grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.max_entry = ttk.Entry(self.tab2)
        self.max_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        # Ввод шага изменения параметра
        self.step_label = ttk.Label(self.tab2, text="Step:")
        self.step_label.grid(row=4, column=0, padx=10, pady=10, sticky="e")
        self.step_entry = ttk.Entry(self.tab2)
        self.step_entry.grid(row=4, column=1, padx=10, pady=10, sticky="w")

        # Кнопка построения бифуркационной диаграммы
        self.bifurcation_button = ttk.Button(self.tab2, text="Построить бифуркацию", command=self.plot_bifurcation)
        self.bifurcation_button.grid(row=5, column=0, columnspan=2, pady=20)

        # Ввод начальной точки X0
        self.bif_X0_label = ttk.Label(self.tab2, text="Начальная точка X0:")
        self.bif_X0_label.grid(row=6, column=0, padx=10, pady=10, sticky="e")
        self.bif_X0_entry = ttk.Entry(self.tab2)
        self.bif_X0_entry.grid(row=6, column=1, padx=10, pady=10, sticky="w")

        # Место для графика
        fig2, self.ax2 = plt.subplots()
        self.canvas2 = FigureCanvasTkAgg(fig2, self.tab2)
        self.canvas2.get_tk_widget().grid(row=7, column=0, columnspan=3, pady=10)
    def setup_controls(self, tab):
        P_label = ttk.Label(tab, text="P:")
        P_label.grid(row=0, column=0, padx=10, pady=10)
        self.P_entry = ttk.Entry(tab)
        self.P_entry.grid(row=0, column=1, padx=10, pady=10)

        A_label = ttk.Label(tab, text="A:")
        A_label.grid(row=1, column=0, padx=10, pady=10)
        self.A_entry = ttk.Entry(tab)
        self.A_entry.grid(row=1, column=1, padx=10, pady=10)

        e_label = ttk.Label(tab, text="e:")
        e_label.grid(row=2, column=0, padx=10, pady=10)
        self.e_entry = ttk.Entry(tab)
        self.e_entry.grid(row=2, column=1, padx=10, pady=10)

        R_label = ttk.Label(tab, text="R:")
        R_label.grid(row=3, column=0, padx=10, pady=10)
        self.R_entry = ttk.Entry(tab)
        self.R_entry.grid(row=3, column=1, padx=10, pady=10)

        Tau0_label = ttk.Label(tab, text="Tau0:")
        Tau0_label.grid(row=4, column=0, padx=10, pady=10)
        self.Tau0_entry = ttk.Entry(tab)
        self.Tau0_entry.grid(row=4, column=1, padx=10, pady=10)

        X0_label = ttk.Label(tab, text="X0:")
        X0_label.grid(row=5, column=0, padx=10, pady=10)
        self.X0_entry = ttk.Entry(tab)
        self.X0_entry.grid(row=5, column=1, padx=10, pady=10)

        X_dot0_label = ttk.Label(tab, text="X_dot0:")
        X_dot0_label.grid(row=6, column=0, padx=10, pady=10)
        self.X_dot0_entry = ttk.Entry(tab)
        self.X_dot0_entry.grid(row=6, column=1, padx=10, pady=10)

        step_label = ttk.Label(tab, text="Step:")
        step_label.grid(row=7, column=0, padx=10, pady=10)
        self.step_entry = ttk.Entry(tab)
        self.step_entry.grid(row=7, column=1, padx=10, pady=10)

        plot_button = ttk.Button(tab, text="Построить", command=self.plot)
        plot_button.grid(row=8, column=0, columnspan=2, pady=20)

    def ask_parameter(self):
        self.parameter = simpledialog.askstring("Выбор параметра")

        return self.parameter

    def plot(self):
        try:
            P = float(self.P_entry.get()) if self.P_entry.get() else 1.0
            A_val = float(self.A_entry.get()) if self.A_entry.get() else 1.0
            e = float(self.e_entry.get()) if self.e_entry.get() else 1.0
            R = float(self.R_entry.get()) if self.R_entry.get() else -1.0
            Tau0 = float(self.Tau0_entry.get()) if self.Tau0_entry.get() else 10
            X0 = float(self.X0_entry.get()) if self.X0_entry.get() else 0
            X_dot0 = float(self.X_dot0_entry.get()) if self.X_dot0_entry.get() else 0
            step = float(self.step_entry.get()) if self.step_entry.get() else 0.01

            # Преобразуем A_val обратно в функцию A для использования в simulate_system
            A = lambda t: A_val * np.sin(t)

            sol = self.simulate_system(P, A, e, R, Tau0, X0, X_dot0, step)
            self.plot_trajectory(sol)
        except ValueError as e:
            tk.messagebox.showerror("Ошибка ввода",
                                    "Пожалуйста, убедитесь, что все поля заполнены корректными числовыми значениями")
            return

    def dynamic_system(self, t, y, P, A_func, e, R):
        x, x_dot = y
        A_value = A_func(t)  # Получаем актуальное значение функции
        x_double_dot = P * A_value - e * np.sin(t)
        return [x_dot, x_double_dot]

    def plate_hit_event(self, t, y):
        x, x_dot = y
        # Событие происходит, когда |x| достигает 1
        return 1 - abs(x)

    plate_hit_event.terminal = True  # Остановить интегрирование при достижении |x| = 1
    plate_hit_event.direction = 0  # Событие срабатывает в обоих направлениях

    def simulate_system(self, P, A, e, R, Tau0, X0, X_dot0, step):
        t_span = [0, Tau0 * 10]
        y0 = [X0, X_dot0]
        sol = solve_ivp(
            fun=lambda t, y: self.dynamic_system(t, y, P, A, e, R),
            t_span=t_span,
            y0=y0,
            events=self.plate_hit_event,
            dense_output=True,
            max_step=step
        )

        # Проверяем, есть ли события столкновения
        while sol.t_events[0].size > 0:
            # Получаем время и состояние последнего события столкновения
            idx = np.searchsorted(sol.t, sol.t_events[0][-1])
            # Обновляем начальные условия для нового интегрирования
            y0 = [np.sign(sol.y[0, idx]) * (1 - 1e-10),
                  -R * sol.y[1, idx]]  # Умножаем скорость на -R и немного отодвигаем от границы
            t_span = [sol.t_events[0][-1], Tau0 * 50]

            # Интегрируем систему с новыми начальными условиями
            new_sol = solve_ivp(
                fun=lambda t, y: self.dynamic_system(t, y, P, A, e, R),
                t_span=t_span,
                y0=y0,
                events=self.plate_hit_event,
                dense_output=True,
                max_step=step
            )

            # Объединяем новые результаты с предыдущими
            sol.t = np.hstack((sol.t, new_sol.t))
            sol.y = np.hstack((sol.y, new_sol.y))
            sol.t_events[0] = np.hstack((sol.t_events[0], new_sol.t_events[0]))

            # Если новое решение не имеет событий столкновения, прерываем цикл
            if new_sol.t_events[0].size == 0:
                break

        return sol

    def plot_trajectory(self, sol):
        # Используем ключи словаря для доступа к данным
        t = sol['t']
        y = sol['y'][0]  # Получаем первую строку y для x

        self.ax1.clear()  # Используйте self.ax1 вместо self.ax
        self.ax1.plot(t, y, label='X(t)')
        self.ax1.set_title('Траектория частицы во времени')
        self.ax1.set_xlabel('Время')
        self.ax1.set_ylabel('X')
        self.ax1.legend()
        self.ax1.grid(True)

        self.ax1.set_ylim([-1, 1])

        self.canvas1.draw()

    def simulate_system_for_R(self, P, A, e, Tau0, X0, X_dot0, step, R_values):
        # Список для сохранения точек бифуркации для каждого значения R
        bifurcation_points = []

        # Определение направления пластины
        selected_plate = self.plate_combobox.get()
        plate_direction = 1 if selected_plate == "Верхняя" else -1

        # Перебор значений параметра R
        for R in R_values:
            t_span = [0, Tau0 * 10]
            y0 = [X0, X_dot0]

            # Интегрирование системы для данного значения R
            sol = solve_ivp(
                fun=lambda t, y: self.dynamic_system(t, y, P, A, e, R),
                t_span=t_span,
                y0=y0,
                events=self.plate_hit_event,
                dense_output=True,
                max_step=step
            )

            if sol.t_events[0].size > 0:
                # Выбор ударов о выбранную пластину и запись бифуркационных точек
                hits = [t for t, x in zip(sol.t_events[0], sol.y[0]) if np.sign(x) == plate_direction]
                hits = hits[int(len(hits) * 0.5):]  # Используем только последние значения
                for t in hits:
                    x_val = sol.sol(t)[0]
                    bifurcation_points.append((R, x_val))

        return bifurcation_points

    def plot_bifurcation(self):
        # Получение параметров из интерфейса
        selected_plate = self.plate_combobox.get()
        parameter = self.parameter_combobox.get()
        if selected_plate not in ["Верхняя", "Нижняя"]:
            messagebox.showerror("Ошибка ввода", "Выберите пластину.")
            return
        if parameter not in ["A", "P", "e", "R"]:
            messagebox.showerror("Ошибка ввода", "Выберите параметр для бифуркации.")
            return

        try:
            min_val = float(self.min_entry.get())
            max_val = float(self.max_entry.get())
            step_val = float(self.step_entry.get())
            bif_X0 = float(self.bif_X0_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка ввода",
                                 "Введите все значения для минимума, максимума, шага и начальной точки X0.")
            return

        P = float(self.P_entry.get()) if self.P_entry.get() else 1.0
        A_val = float(self.A_entry.get()) if self.A_entry.get() else 1.0
        e = float(self.e_entry.get()) if self.e_entry.get() else 1.0
        R = float(self.R_entry.get()) if self.R_entry.get() else 1.0
        Tau0 = float(self.Tau0_entry.get()) if self.Tau0_entry.get() else 10.0
        X_dot0 = float(self.X_dot0_entry.get()) if self.X_dot0_entry.get() else 0.0
        step = float(self.step_entry.get()) if self.step_entry.get() else 0.01

        # Определение направления пластины
        plate_direction = 1 if selected_plate == "Верхняя" else -1

        # Установка A_func по умолчанию
        A_func = lambda t: A_val * np.sin(t)

        bifurcation_points = []

        if parameter == 'R':
            R_values = np.arange(min_val, max_val, step_val)
            bifurcation_points = self.simulate_system_for_R(P, A_func, e, Tau0, bif_X0, X_dot0, step, R_values)
        else:
            # Для параметров A, P и e используем обычный цикл
            val_range = np.arange(min_val, max_val, step_val)
            for val in val_range:
                if parameter == 'A':
                    A_func = lambda t, val=val: val * np.sin(t)
                elif parameter == 'P':
                    P = val
                elif parameter == 'e':
                    e = val

                sol = self.simulate_system(P, A_func, e, R, Tau0, bif_X0, X_dot0, step)

                # Проверяем, есть ли события столкновения
                if sol.t_events and sol.t_events[0].size > 0:
                    # Выбор ударов о выбранную пластину
                    hits = [t for t, x in zip(sol.t_events[0], sol.y[0]) if np.sign(x) == plate_direction]
                    # Используем последние значения после преходного процесса
                    hits = hits[int(len(hits) * 0.5):]
                    for t in hits:
                        x_val = sol.sol(t)[0]
                        bifurcation_points.append((val, x_val))

        # Построение бифуркационной диаграммы
        if bifurcation_points:
            values, xs = zip(*bifurcation_points)
            self.ax2.clear()
            self.ax2.plot(values, xs, '.k', markersize=1)
            self.ax2.set_title(f"Бифуркационная диаграмма для параметра {parameter}")
            self.ax2.set_xlabel("Значение параметра")
            self.ax2.set_ylabel("X")
            self.canvas2.draw()
        else:
            messagebox.showinfo("Информация", "Нет данных для отображения бифуркационной диаграммы.")
def main():
    root = tk.Tk()
    app = NuclearExplosionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
