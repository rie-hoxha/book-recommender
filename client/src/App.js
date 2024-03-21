import React, { useState } from 'react';
import { Input, Button, List, Card } from 'antd';
import { UserOutlined, RobotOutlined } from '@ant-design/icons';

function App() {
  const [inputValue, setInputValue] = useState('');
  const [recommendations, setRecommendations] = useState([]);

  const fetchRecommendations = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/recommend/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputValue }),
      });
      const data = await response.json();
      if (data.response) {
        setRecommendations([data.response]); // Assuming response is a single book object
      }
    } catch (error) {
      console.error('Error fetching recommendations:', error);
    }
  };

  return (
    <div style={{ padding: '50px' }}>
      <Card title="Book Recommender Chatbot">
        <Input
          prefix={<UserOutlined />}
          placeholder="Ask for a book recommendation..."
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          style={{ marginBottom: '20px' }}
        />
        <Button type="primary" onClick={fetchRecommendations}>
          Ask
        </Button>
        <List
          itemLayout="horizontal"
          dataSource={recommendations}
          renderItem={item => (
            <List.Item>
              <List.Item.Meta
                avatar={<RobotOutlined />} // Robot icon as avatar
                title={<a href="https://ant.design">{item.Book}</a>}
                description={item.Feedback}
              />
              <div>Rating: {item.Rate}</div>
            </List.Item>
          )}
        />
      </Card>
    </div>
  );
}


export default App;
