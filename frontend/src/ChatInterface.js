import React, { useState } from 'react';

function ChatInterface({ messages, onSend }) {
  const [input, setInput] = useState('');

  const handleSend = async () => {
    if (input.trim()) {
      onSend(input);
      setInput('');
    }
  };

  return (
    <div>
      <div>
        {messages.map((msg, index) => (
          <div key={index}>
            <strong>{msg.user ? 'User' : 'Bot'}:</strong> {msg.text}
          </div>
        ))}
      </div>
      <input value={input} onChange={(e) => setInput(e.target.value)} />
      <button onClick={handleSend}>Send</button>
    </div>
  );
}

export default ChatInterface;
