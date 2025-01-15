import { HistoryResponse, HistoryStats } from '@/types/history'

const token = 'your_auth_token_here' // Replace with actual token management

export const historyApi = {
  getHistory: async (page: number = 1, perPage: number = 10, type?: string): Promise<HistoryResponse> => {
    const params = new URLSearchParams({
      page: page.toString(),
      per_page: perPage.toString(),
      ...(type && { type })
    });
    try {
      const response = await fetch(`https://hypercover.replit.app/api/history?${params}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error("Failed to fetch history:", error);
      throw new Error('Failed to fetch history. Please try again later.');
    }
  },

  addHistory: async (activity_type: string, details: Record<string, any> = {}): Promise<void> => {
    const response = await fetch('https://hypercover.replit.app/api/history', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ activity_type, details })
    });
    if (!response.ok) {
      throw new Error('Failed to add history entry');
    }
  },

  clearHistory: async (): Promise<void> => {
    const response = await fetch('https://hypercover.replit.app/api/history/clear', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    if (!response.ok) {
      throw new Error('Failed to clear history');
    }
  },

  getStats: async (): Promise<HistoryStats> => {
    const response = await fetch('https://hypercover.replit.app/api/history/stats', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    if (!response.ok) {
      throw new Error('Failed to fetch history stats');
    }
    return response.json();
  }
};

