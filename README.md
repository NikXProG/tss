# Библиотека TSS

**TSS** (Thomas Solver System) — это коллекция Python-модулей для решения различных задач в области численных методов, дифференциальных уравнений и физического моделирования. Она включает:

1. **Thomas Algorithm** — эффективный метод решения тридиагональных систем линейных уравнений (используется в численном анализе).
2. **Grid Function Generator** — генератор сеточных функций с различными математическими функциями (экспонента, синус, косинус и т.д.).
3. **BVP Solver** — решатель краевых задач (boundary value problems) с использованием scipy.
4. **Spring-Mass-Damper** — модель пружинно-массовой системы с демпфированием (физическое моделирование).
5. **Derivative Comparison** — сравнение численных и аналитических производных.
6. **Euler System** — метод Эйлера для решения систем обыкновенных дифференциальных уравнений (ОДУ).

Библиотека предназначена для образовательных целей, научных расчётов и демонстрации численных методов.

## Установка и настройка

1. **Установка зависимостей**:
   ```bash
   pip install -e .
   ```
   Это установит библиотеку в режиме разработки и все зависимости (numpy, matplotlib, scipy).

2. **Запуск всех демонстраций**:
   ```bash
   python main.py
   ```
   Это запустит последовательную демонстрацию всех модулей с графиками и выводами.

3. **Запуск тестов**:
   ```bash
   python -m pytest tests/
   ```
   Или через taskipy: `python -m taskipy.cli test`

## Как пользоваться библиотекой

### 1. Импорт и использование модулей

#### Thomas Algorithm (решение тридиагональных систем)
```python
from tss import TridiagonalSolver

# Определите коэффициенты тридиагональной матрицы
a = [-1, -1]  # нижняя диагональ
b = [2, 2, 2]  # главная диагональ
c = [-1, -1]  # верхняя диагональ
d = [1, 0, 1]  # правая часть

solver = TridiagonalSolver(a, b, c, d)
solution = solver.solve()
residuals = solver.residual(solution)

print("Решение:", solution)
print("Остатки:", residuals)
```

#### Grid Function Generator (генерация сеточных функций)
```python
from tss import GridFunctionGenerator

gen = GridFunctionGenerator()

# Генерация сетки и значений функции
x_grid, y_grid = gen.generate_grid_function(
    func=lambda x: np.sin(x),
    a=0, b=2*np.pi, h=0.1
)

# Создание мелкой сетки для гладких графиков
x_fine = gen.create_fine_grid(0, 2*np.pi)
```

#### BVP Solver (краевые задачи)
```python
from tss.bvp_solver import BoundaryValueProblem

bvp = BoundaryValueProblem()
sol = bvp.solve_bvp(n_points=500)

if sol.success:
    print("Решение найдено!")
    # Доступ к решению: sol.sol(x) для значений в точке x
```

#### Spring-Mass-Damper (пружинно-массовая система)
```python
from tss.spring_mass_damper import DampedSpringMass

system = DampedSpringMass(m=1.0, k=2.0)
system.plot_solutions(t_span=(0, 20), v0=1.0, cases=[1, 2, 3, 4])
```

#### Derivative Comparison (сравнение производных)
```python
from tss.derivative_comparison import plot_derivative_comparison

plot_derivative_comparison()  # Строит графики сравнения
```

#### Euler System (метод Эйлера для систем ОДУ)
```python
from tss.euler_system import plot_euler_solutions

plot_euler_solutions()  # Решает и строит графики для примеров систем
```

### 2. Структура проекта

- `tss/` — папка с модулями библиотеки
- `main.py` — скрипт для запуска всех демонстраций
- `tests/` — unit-тесты
- `pyproject.toml` — конфигурация проекта (зависимости, инструменты)
- `README.md` — документация

### 3. Качество кода

Проект использует:
- **Ruff** для линтинга и форматирования
- **Pytest** для тестирования
- **Taskipy** для автоматизации задач

Запуск проверки качества:
```bash
./run.sh  # или python -m taskipy.cli check
```

## Примеры применения

- **Образование**: демонстрация численных методов студентам
- **Наука**: быстрые расчёты тридиагональных систем, решение ОДУ
- **Инженерия**: моделирование физических систем (пружина-демпфер)
- **Разработка**: как основа для более сложных симуляций

## Лицензия

MIT License
