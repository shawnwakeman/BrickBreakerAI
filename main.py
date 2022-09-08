import os.path
import pygame
import neat
import sys
import random
import pickle

pygame.init()


WIN = pygame.display.set_mode((800, 600))
class Paddle:

    def __init__(self):
        self.x = 300
        self.y = 200
        self.speed = 10

    def manual_imput(self):
        key = pygame.key.get_pressed()
        if key[pygame.K_LEFT]:
            self.move_left()
        if key[pygame.K_RIGHT]:
            self.move_right()

    def move_left(self):
        if self.x > 0:
            self.x -= 10

    def move_right(self):
        if self.x < 700:
            self.x += 10

class Ball:

    def __init__(self):
        self.x = 400
        self.y = 450
        self.x_vel = 3
        self.y_vel = 6

    def movement(self):
        if self.x <= 0 and self.y >= 0 and self.y <= 588:
            self.x_vel *= -1
        if self.x >= 788 and self.y >= 0 and self.y <= 588:
            self.x_vel *= -1
        if self.y <= 0 and self.x >= 0 and self.x <= 788:
            self.y_vel *= -1
        self.x += self.x_vel
        self.y += self.y_vel


class Game:

    def __init__(self):
        self.WIDTH = 800
        self.HEIGHT = 600
        self.FONT = pygame.font.SysFont("Bebas Neue", 42)
        self.paddle = Paddle()
        self.ball = Ball()
        self.bricks_list = []
        self.paddle_hits = 0
        for y in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
            for x in [0,1,2,3,4,5,6,7,8,9,10,10]:
                tup = (x * 80 + 10, y * 20)
                if y % 2 == 0:
                    if x % 2 ==0:
                        self.bricks_list.append(tup)

                else:
                    if x % 2 !=0:
                        self.bricks_list.append(tup)




    def loop(self, draw = True):
        WIN.fill((100,100,100))
        self.paddle.manual_imput()

        self.ball.movement()
        self.paddle_blit = pygame.draw.rect(WIN, (150,150,150), pygame.Rect(self.paddle.x, 550, 100, 25))
        self.ball_blit = pygame.draw.rect(WIN, (150, 150, 150), pygame.Rect(self.ball.x, self.ball.y, 12, 12))


        if self.ball_blit.colliderect(self.paddle_blit):
            self.ball.y = 535
            self.ball.y_vel *= -1
            self.paddle_hits += 1


        for i,b in enumerate(self.bricks_list):
            self.brick = pygame.draw.rect(WIN, (255,150,150), pygame.Rect(b[0], b[1], 60, 10))
            if self.ball_blit.colliderect(self.brick):
                self.ball.y_vel *= -1
                self.bricks_list.pop(i)



def train_ai(genome1, config):
    run = True
    clock = pygame.time.Clock()
    game = Game()

    net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
    while run:
        key = pygame.key.get_pressed()
        if key[pygame.K_6]:
            clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
        output1 = net1.activate((game.ball.x, game.ball.y, game.paddle.x, game.paddle_hits))
        decision1 = output1.index(max(output1))
        if decision1 == 0:
            game.paddle.move_left()
        elif decision1 == 1:
            game.paddle.move_right()

        if game.ball.y >= 588 and game.ball.x >= 0 and game.ball.x <= 788:
            genome1.fitness += game.paddle_hits
            break
        game.loop()
        pygame.display.update()



def eval_genomes(genomes, config):

    for i, (genome_id1,genome1) in enumerate(genomes):
        genome1.fitness = 0
        train_ai(genome1, config)



def play_best(config):

    run = True
    game = Game()
    clock = pygame.time.Clock()

    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)


    while run:
        clock.tick(60)
        key = pygame.key.get_pressed()
        if key[pygame.K_6]:
            clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
        output1 = winner_net.activate((game.ball.x, game.ball.y, game.paddle.x, game.paddle_hits))
        decision1 = output1.index(max(output1))
        if decision1 == 0:
            game.paddle.move_left()
        elif decision1 == 1:
            game.paddle.move_right()

        if game.ball.y >= 588 and game.ball.x >= 0 and game.ball.x <= 788:
            play_best(config)
        game.loop()
        pygame.display.update()



        game.loop()

        pygame.display.update()
        key = pygame.key.get_pressed()
        if key[pygame.K_6]:
            break

def run_neat(config):
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-85')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)


    winner = p.run(eval_genomes,100)

    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)
    play_best(config)