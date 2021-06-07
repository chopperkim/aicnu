import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 100  # pixels
HEIGHT = 5  # grid height
WIDTH = 5  # grid width


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('monte carlo')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []



    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                                height=HEIGHT * UNIT,
                                width=WIDTH * UNIT)
        # create grids
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        # add img to canvas
        self.dog = canvas.create_image(50, 50, image=self.shapes[0])
        self.rock_1 = canvas.create_image(250, 150, image=self.shapes[1])
        self.rock_2 = canvas.create_image(150, 250, image=self.shapes[1])
        self.circle    = canvas.create_image(250, 250, image=self.shapes[2])

        # pack all
        canvas.pack()

        return canvas

    def load_images(self):
        dog  = PhotoImage( Image.open("./img/dog.png").resize((65, 65)) )
        rock = PhotoImage( Image.open("./img/rock.png").resize((65, 65)) )
        meat = PhotoImage( Image.open("./img/meat.png").resize((65, 65)) )

        return dog, rock, meat

    @staticmethod
    def coords_to_state(coords):
        x = int((coords[0] - 50) / 100)
        y = int((coords[1] - 50) / 100)
        return [x, y]

    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.dog)
        self.canvas.move(self.dog, UNIT / 2 - x, UNIT / 2 - y)
        # return observation
        return self.coords_to_state(self.canvas.coords(self.dog))

    def step(self, action, value_table):
        state = self.canvas.coords(self.dog)
        base_action = np.array([0, 0])
        self.render()

        if action == 0:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # right
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        # move agent
        self.canvas.move(self.dog, base_action[0], base_action[1])
        # move rectangle to top level of canvas
        self.canvas.tag_raise(self.dog)

        next_state = self.canvas.coords(self.dog)

        # reward function
        
        # meat!
        if next_state == self.canvas.coords(self.circle):
            reward = 100 
            done = True
        # rock
        elif next_state in [self.canvas.coords(self.rock_1),
                            self.canvas.coords(self.rock_2)]:
            reward = -100
            done = True
        # other state
        else:
            reward = 0
            done = False

        next_state = self.coords_to_state(next_state)

        return next_state, reward, done

    def print_values(self, values):
        # clear canvas
        for i in self.texts:
            self.canvas.delete(i)

        # update display
        for i in range(WIDTH):
            for j in range(HEIGHT):
                coord_key = '[{}, {}]'.format(i,j)
                self.text_value(i, j, values[coord_key])
        self.update() # ui update


    def text_value(self, col, row, contents, font='Helvetica', size=12, style='normal', anchor="nw"):
        origin_x, origin_y = 85, 70
        x, y = origin_y + (UNIT * col) -25.0, origin_x + (UNIT * row)
        font = (font, str(size), style)

        contents = round(contents, 4)
        text = self.canvas.create_text(x, y, fill="black", text=contents, font=font, anchor=anchor)
        return self.texts.append(text)

    def render(self):
        # print values
        time.sleep(0.05)
        self.update() # ui update

        
