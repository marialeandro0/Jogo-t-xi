import gymnasium as gym  # Importando biblioteca gym
import numpy as np
import pygame  # Importando o Pygame

# Inicializando o pygame
pygame.init()

# Configurações da tela do Pygame
screen_width, screen_height = 600, 400
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Renderização com Pygame")

# Criação do ambiente
env = gym.make('Taxi-v3', render_mode='human')
env.reset()

Q = np.zeros([env.observation_space.n, env.action_space.n])  # Inicialização da tabela Q

# Definindo parâmetros para aprendizado Q
alpha = 0.1  # Taxa de aprendizado
gamma = 0.99  # Fator de desconto
epsilon = 0.1  # Taxa de exploração
episodes = 1000  # Número de episódios de treinamento

# Função para renderizar o estado do ambiente usando Pygame
def render_with_pygame(state):
    # Limpa a tela
    screen.fill((0, 0, 0))  # Fundo preto
    
    # Representação simples do estado no pygame (Aqui você pode melhorar com imagens ou gráficos)
    font = pygame.font.SysFont('Arial', 24)
    state_text = font.render(f"Estado: {state}", True, (255, 255, 255))
    screen.blit(state_text, (10, 10))

    # Atualiza a tela do Pygame
    pygame.display.flip()

# Treinamento do agente usando a tabela Q
for episode in range(episodes):
    state, _ = env.reset()  # Reseta o ambiente
    done = False
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Escolher uma ação aleatória
        else:
            action = np.argmax(Q[state])  # Escolher a ação com o maior valor Q
        
        # Realiza a ação no ambiente
        next_state, reward, done, _, _ = env.step(action)
        
        # Atualizando a tabela Q
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        # Atualiza o estado
        state = next_state
        
        # Renderizando com Pygame
        render_with_pygame(state)

# Testando desempenho do agente
state, _ = env.reset()  # Reseta o ambiente e obtém o estado inicial
done = False
total_rewards = 0

while not done:
    action = np.argmax(Q[state])  # Agente escolhe a melhor ação com base na Q-table
    state, reward, done, _, _ = env.step(action)
    total_rewards += reward
    env.render()
    
    # Renderizando com Pygame
    render_with_pygame(state)

print(f"Recompensa total: {total_rewards}")
env.close()
pygame.quit()  # Finaliza o Pygame