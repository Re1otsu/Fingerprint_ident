import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from finger import (
    register_employee,
    verify_employee,
    list_employees,
    list_logs,
    init_db,
)


class AdminGUI(tk.Tk):  
    def __init__(self):
        super().__init__()

        self.title("Fingerprint Access Control - Admin")
        self.geometry("800x600")

        init_db()

        self.selected_image = None

        self.create_widgets()
        self.refresh_employees()
        self.refresh_logs()

    def create_widgets(self):
        # Верхняя панель ввода
        frame_top = tk.Frame(self)
        frame_top.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(frame_top, text="Employee ID:").grid(row=0, column=0, sticky="w")
        self.emp_id_var = tk.StringVar()
        tk.Entry(frame_top, textvariable=self.emp_id_var, width=20).grid(row=0, column=1, padx=5)

        tk.Button(frame_top, text="Выбрать файл отпечатка", command=self.choose_file)\
            .grid(row=0, column=2, padx=5)

        tk.Button(frame_top, text="Зарегистрировать", command=self.on_register)\
            .grid(row=0, column=3, padx=5)

        tk.Button(frame_top, text="Проверить", command=self.on_verify)\
            .grid(row=0, column=4, padx=5)

        self.file_label_var = tk.StringVar(value="Файл не выбран")
        tk.Label(frame_top, textvariable=self.file_label_var, fg="gray")\
            .grid(row=1, column=0, columnspan=5, sticky="w", pady=5)

        # Разделитель
        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, padx=10, pady=5)

        # Средняя часть: список сотрудников
        frame_mid = tk.Frame(self)
        frame_mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Список сотрудников
        frame_emp = tk.LabelFrame(frame_mid, text="Посетители")
        frame_emp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.emp_tree = ttk.Treeview(frame_emp, columns=("id", "created"), show="headings")
        self.emp_tree.heading("id", text="Employee ID")
        self.emp_tree.heading("created", text="Created At (UTC)")
        self.emp_tree.column("id", width=100)
        self.emp_tree.column("created", width=200)
        self.emp_tree.pack(fill=tk.BOTH, expand=True)

        # Логи
        frame_logs = tk.LabelFrame(frame_mid, text="Логи доступа (последние)")
        frame_logs.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.logs_text = tk.Text(frame_logs, height=20)
        self.logs_text.pack(fill=tk.BOTH, expand=True)

        # Нижняя панель
        frame_bottom = tk.Frame(self)
        frame_bottom.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(frame_bottom, text="Обновить список", command=self.refresh_all)\
            .pack(side=tk.RIGHT)

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Выберите файл отпечатка",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if path:
            self.selected_image = path
            self.file_label_var.set(path)

    def on_register(self):
        emp_id = self.emp_id_var.get().strip()
        if not emp_id:
            messagebox.showerror("Ошибка", "Введите Employee ID")
            return
        if not self.selected_image:
            messagebox.showerror("Ошибка", "Выберите файл отпечатка")
            return

        try:
            register_employee(emp_id, self.selected_image)
            messagebox.showinfo("Успех", f"Посетитель {emp_id} уже был зарегистрирован.")
            self.refresh_all()
        except Exception as e:
            messagebox.showerror("Ошибка регистрации", str(e))

    def on_verify(self):
        emp_id = self.emp_id_var.get().strip()
        if not emp_id:
            messagebox.showerror("Ошибка", "Введите Employee ID")
            return
        if not self.selected_image:
            messagebox.showerror("Ошибка", "Выберите файл отпечатка")
            return

        try:
            ok = verify_employee(emp_id, self.selected_image)
            if ok:
                messagebox.showinfo("Доступ", "ДОСТУП РАЗРЕШЁН")
            else:
                messagebox.showwarning("Доступ", "ДОСТУП ЗАПРЕЩЁН")
            self.refresh_logs()
        except Exception as e:
            messagebox.showerror("Ошибка проверки", str(e))

    def refresh_employees(self):
        for row in self.emp_tree.get_children():
            self.emp_tree.delete(row)
        for emp_id, created_at in list_employees():
            self.emp_tree.insert("", tk.END, values=(emp_id, created_at))

    def refresh_logs(self):
        self.logs_text.delete("1.0", tk.END)
        rows = list_logs(limit=100)
        for emp_id, ts, result in rows:
            self.logs_text.insert(tk.END, f"[{ts}] {emp_id}: {result}\n")

    def refresh_all(self):
        self.refresh_employees()
        self.refresh_logs()


if __name__ == "__main__":
    app = AdminGUI()
    app.mainloop()
