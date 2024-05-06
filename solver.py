import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     

    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]] 
    temp = []
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    return temp

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = collections.deque([[0]])
    temp = []
    ### CODING FROM HERE ###

def cost(actions):
    """A cost function"""
    return len([x for x in actions if isinstance(x, str) and x.islower()])

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    start = time.time() #Ghi lại thời gian bắt đầu
    beginBox = PosOfBoxes(gameState) #Lấy vị trí ban đầu của các hộp
    beginPlayer = PosOfPlayer(gameState) #Lấy vị trí ban đầu của người chơi

    startState = (beginPlayer, beginBox) #Khởi tạo trạng thái ban đầu là một tuple gồm vị trí người chơi và vị trí các hộp
    frontier = PriorityQueue() #Tạo một hàng đợi ưu tiên frontier
    frontier.push([startState], 0) #Thêm trạng thái ban đầu vào frontier với ưu tiên 0
    exploredSet = set() #Tạo một tập rỗng để lưu trữ các trạng thái đã được duyệt qua
    actions = PriorityQueue() #Tạo một hàng đợi ưu tiên actions
    actions.push([0], 0) #Thêm trạng thái ban đầu vào actions với ưu tiên 0
    temp = [] #Tạo một list rỗng để lưu trữ lịch sử các hành động dẫn đến mục tiêu
    sonutmora = 0 #Tạo biến đếm số nút được mở ra
    ### CODING FROM HERE ###
    while frontier: #Lặp khi frontier không rỗng
        node = frontier.pop() #Lấy ra trạng thái có ưu tiên cao nhất từ frontier
        node_action = actions.pop() #Lấy ra danh sách hành động tương ứng với trạng thái hiện tại
        if isEndState(node[-1][-1]): #Kiểm tra xem trạng thái hiện tại có phải là trạng thái đích hay không
            temp += node_action[1:] #Nếu là trạng thái đích, thêm các hành động từ vị trí thứ 2 vào temp
            break #Thoát khỏi vòng lặp
        if node[-1] not in exploredSet: #Kiểm tra xem trạng thái hiện tại đã được duyệt chưa
            exploredSet.add(node[-1]) #Nếu chưa, thêm trạng thái vào tập exploredSet
            sonutmora += 1 #Tăng dần giá trị biến đếm khi có 1 nút mới được mở ra
            for action in legalActions(node[-1][0], node[-1][1]): #Lấy danh sách các hành động hợp lệ từ trạng thái hiện tại
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) #Áp dụng hành động để tính toán trạng thái mới
                if isFailed(newPosBox): #Kiểm tra xem trạng thái mới có bị failed hay không
                    continue #Nếu có, bỏ qua trạng thái này
                new_node = node + [(newPosPlayer, newPosBox)] #Tạo trạng thái mới
                new_action = node_action + [action[-1]] #Tạo danh sách hành động mới
                frontier.push(new_node, cost(new_action)) #Thêm trạng thái mới vào frontier với ưu tiên là chi phí của danh sách hành động mới
                actions.push(new_action, cost(new_action)) #Thêm danh sách hành động mới vào actions với ưu tiên là chi phí của nó
    end = time.time() #Ghi lại thời gian kết thúc
    print(f'Thoi gian chay: {end - start:.3f} giay') #In ra thời gian chạy của thuật toán
    print(f'So nut da duoc mo ra: {sonutmora}') #In ra số nút đã được mở ra
    return temp #Trả về chuỗi các hành động        

def heuristic(posPlayer, posBox):
    # print(posPlayer, posBox)
    """A heuristic function to calculate the overall distance between the else boxes and the else goals"""
    distance = 0
    completes = set(posGoals) & set(posBox)
    sortposBox = list(set(posBox).difference(completes))
    sortposGoals = list(set(posGoals).difference(completes))
    for i in range(len(sortposBox)):
        distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + (abs(sortposBox[i][1] - sortposGoals[i][1]))
    return distance

def aStarSearch(gameState):
    """Implement aStarSearch approach"""
    start =  time.time() #Ghi lại thời gian bắt đầu
    beginBox = PosOfBoxes(gameState) #Lấy vị trí ban đầu của các hộp
    beginPlayer = PosOfPlayer(gameState) #Lấy vị trí ban đầu của người chơi
    temp = [] #Tạo một list rỗng để lưu trữ lịch sử các hành động dẫn đến mục tiêu
    start_state = (beginPlayer, beginBox) #Khởi tạo trạng thái ban đầu là một tuple gồm vị trí người chơi và vị trí các hộp
    frontier = PriorityQueue() #Tạo một hàng đợi ưu tiên frontier
    frontier.push([start_state], heuristic(beginPlayer, beginBox)) #Thêm trạng thái ban đầu vào frontier với ưu tiên là giá trị hàm heuristic
    exploredSet = set() #Tạo một tập rỗng để lưu trữ các trạng thái đã được duyệt qua
    actions = PriorityQueue() #Tạo một hàng đợi ưu tiên actions
    actions.push([0], heuristic(beginPlayer, start_state[1])) #Thêm hành động rỗng vào actions với ưu tiên là giá trị hàm heuristic
    sonutmora = 0 #Tạo biến đếm số nút được mở ra
    while len(frontier.Heap) > 0: #Lặp khi frontier không rỗng
        node = frontier.pop() #Lấy ra trạng thái có ưu tiên cao nhất từ frontier
        node_action = actions.pop() #Lấy ra danh sách hành động tương ứng với trạng thái hiện tại
        if isEndState(node[-1][-1]): #Kiểm tra xem trạng thái hiện tại có phải là trạng thái đích hay không
            temp += node_action[1:] #Nếu là trạng thái đích, thêm các hành động từ vị trí thứ 2 vào temp
            break #Thoát khỏi vòng lặp

        ### CONTINUE YOUR CODE FROM HERE
        exploredSet.add(node[-1]) #Thêm trạng thái hiện tại vào tập exploredSet
        sonutmora += 1 #Tăng dần giá trị biến đếm khi có 1 nút mới được mở ra
        for action in legalActions(node[-1][0], node[-1][1]): #Lấy danh sách các hành động hợp lệ từ trạng thái hiện tại
            newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) #Áp dụng hành động để tính toán trạng thái mới
            if isFailed(newPosBox): #Kiểm tra xem trạng thái mới có bị failed hay không
                continue #Nếu có, bỏ qua trạng thái này
            new_state = (newPosPlayer, newPosBox) #Tạo trạng thái mới
            if new_state not in exploredSet: #Kiểm tra xem trạng thái mới đã được duyệt chưa
                new_cost = cost(node_action[1:]) + 1 #Tính chi phí của trạng thái mới
                new_priority = new_cost + heuristic(newPosPlayer, newPosBox) #Tính ưu tiên của trạng thái mới
                frontier.push(node + [new_state], new_priority) #Thêm trạng thái mới vào frontier với ưu tiên tính toán được
                actions.push(node_action + [action[-1]], new_priority) #Thêm hành động mới vào actions với ưu tiên tương ứng

    end =  time.time() #Ghi lại thời gian kết thúc
    print(f'Thoi gian chay: {end - start:.3f} giay') #In ra thời gian chạy của thuật toán
    print(f'So nut da duoc mo ra: {sonutmora}') #In ra số nút đã được mở ra
    return temp #Trả về chuỗi các hành động

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':        
        result = breadthFirstSearch(gameState)
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    elif method == 'astar':
        result = aStarSearch(gameState)        
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)
    return result
