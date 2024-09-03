"use client";

import Image from "next/image";
import { useChat } from "ai/react";
import { useState } from "react";

export default function Home() {
  const { messages, input, handleInputChange, handleSubmit } = useChat();
 
  return (
    <div className="bg-black font-mono text-white">
      <header className="py-4 px-2 max-w-3xl mx-auto">
        <h1 className="text-2xl font-bold mb-1">Mirror</h1>
        <p className="text-sm">
          AI-powered search engine for your computer
        </p>
      </header>

      <div className="flex flex-col w-full max-w-3xl pt-4 pb-[40px] mx-auto">
        <div className="flex flex-col flex-nowrap flex-grow flex-shrink gap-2">
          {messages.map((m) => (
            <div key={m.id} className="whitespace-pre-wrap p-2 text-[15px]">
              {m.role === "user" ? (
                <span className="font-bold text-amber-400">user$ </span>
              ) : (
                <span className="font-bold text-teal-300">mirror$ </span>
              )}
              {m.content}
            </div>
          ))}
        </div>
      </div>
      
      <form onSubmit={handleSubmit} className="fixed bottom-0 w-full flex justify-center mb-2">
        <input
          value={input}
          onChange={handleInputChange}
          placeholder="Explore your memories..."
          className="w-full max-w-3xl p-2 rounded-lg bg-black text-white border border-white text-sm focus:outline-none  transition duration-200"
        />
      </form>
    </div>
  );
}