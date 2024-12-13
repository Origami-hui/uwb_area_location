import mysql.connector

# 连接到 MySQL 数据库
connection = mysql.connector.connect(
    host='localhost',      # 数据库主机
    user='root',           # 数据库用户名
    password='123456',   # 数据库密码
    database='uwb_location'     # 数据库名称
)

cursor = connection.cursor()
# print("数据库初始化！")


def get_database_record():
    # 执行查询
    cursor.execute("SELECT * FROM tag_location ")

    # 获取查询结果
    results = cursor.fetchall()
    for row in results:
        print(row)


def insert(tag_id, tx_location, nlos_state):
    try:
        # 确保 tx_location 是一个包含两个数值的列表或元组
        if len(tx_location) != 2:
            raise ValueError("tx_location must contain exactly two elements: [x, y]")

        # 定义要插入的数据
        params = [tag_id, tx_location[0], tx_location[1], nlos_state]

        # 执行查询，插入数据
        cursor.execute("""
            INSERT INTO tag_location (tag_id, tx_x, tx_y, nlos_state) 
            VALUES (%s, %s, %s, %s)
        """, params)

        # 提交事务
        connection.commit()

    except mysql.connector.Error as err:
        # 处理数据库错误
        print(f"Error: {err}")
        connection.rollback()  # 回滚事务，确保数据一致性

    except Exception as e:
        # 处理其他异常
        print(f"An error occurred: {e}")
        connection.rollback()


def close_dao():
    # 关闭 cursor 和连接
    cursor.close()
    connection.close()


# insert(5, [1, 2], 1)
# get_database_record()
# close()