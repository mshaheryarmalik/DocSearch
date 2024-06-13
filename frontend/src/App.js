// App.js

import React, { useState, useEffect } from 'react';
import FileUpload from './FileUpload';
import ChatInterface from './ChatInterface';

function App() {
  const [messages, setMessages] = useState([]);
  const [threadId, setThreadId] = useState(null); // Initialize threadId state

  useEffect(() => {
    console.log(threadId); // Log the threadId when it changes
  }, [threadId]); // Add threadId as a dependency to the useEffect hook

  const handleUpload = async (newThreadId) => { // Update handleUpload to capture threadId
    setThreadId(newThreadId); // Set the threadId state
    setMessages([]);
  };

  const handleSend = async (message) => {
    setMessages([...messages, { user: true, text: message }]);
    try {
      console.log(threadId);
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threadId, message }), // Pass threadId back to the backend
      });
      const data = await response.json();
      setMessages((prevMessages) => [...prevMessages, { user: false, text: data.response }]);
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  return (
    <div>
      <FileUpload onUpload={handleUpload} /> {/* Pass handleUpload to FileUpload component */}
      <ChatInterface messages={messages} onSend={handleSend} />
    </div>
  );
}

export default App;
