import React from 'react'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

type Endpoint = {
  method: string
  path: string
  description: string
  requestBody?: string
  responses: {
    [key: string]: string
  }
}

type EndpointGroup = {
  name: string
  endpoints: Endpoint[]
}

const endpointGroups: EndpointGroup[] = [
  {
    name: "Authentication",
    endpoints: [
      {
        method: "POST",
        path: "/api/auth/register",
        description: "Register a new user",
        requestBody: `{
  "email": string (email format),
  "password": string (min length: 8)
}`,
        responses: {
          "201": "User registered successfully",
          "400": "Invalid request parameters"
        }
      },
      {
        method: "POST",
        path: "/api/auth/login",
        description: "Login a user",
        requestBody: `{
  "email": string (email format),
  "password": string
}`,
        responses: {
          "200": `{
  "token": string (JWT token)
}`,
          "401": "Invalid credentials"
        }
      }
    ]
  },
  {
    name: "Cover Letter Generation",
    endpoints: [
      {
        method: "POST",
        path: "/api/bulk-generate",
        description: "Generate multiple cover letters",
        requestBody: `{
  "user_profile": string (JSON format - user profile information),
  "template_id": string,
  "session_id": string,
  "file": binary (Excel/CSV file with job descriptions)
}`,
        responses: {
          "200": `{
  "result": [
    {
      "job_id": string,
      "status": "success" | "error",
      "cover_letter": string,
      "error": string
    }
  ]
}`,
          "400": "Invalid request parameters",
          "401": "Unauthorized",
          "429": "Rate limit exceeded",
          "500": "Server error"
        }
      },
      {
        method: "GET",
        path: "/api/templates",
        description: "Get available templates",
        responses: {
          "200": `{
  "templates": [
    {
      "id": string,
      "name": string,
      "description": string,
      "tone_options": [string]
    }
  ]
}`,
          "401": "Unauthorized",
          "429": "Rate limit exceeded"
        }
      }
    ]
  },
  {
    name: "Job Suggestion",
    endpoints: [
      {
        method: "POST",
        path: "/api/resume-suggest",
        description: "Get job suggestions based on resume",
        requestBody: `{
  "resume_data": {
    "format": "pdf" | "docx" | "txt",
    "content": string (base64 encoded)
  },
  "preferences": {
    "location": string (e.g., "San Francisco, CA"),
    "work_type": "remote" | "hybrid" | "onsite" | "any",
    "industry": [string],
    "salary_range": {
      "min": number,
      "max": number,
      "currency": string
    }
  }
}`,
        responses: {
          "200": "Successful response with job suggestions, market insights, and resume analysis",
          "400": "Invalid request parameters",
          "408": "Request timeout",
          "429": "Rate limit exceeded",
          "500": "Server error"
        }
      }
    ]
  },
  {
    name: "Subscription Management",
    endpoints: [
      {
        method: "POST",
        path: "/api/subscription/create-checkout",
        description: "Create a checkout session for subscription",
        requestBody: `{
  "price_id": string (Stripe price ID)
}`,
        responses: {
          "200": `{
  "session_id": string
}`,
          "401": "Unauthorized"
        }
      }
    ]
  },
  {
    name: "History Management",
    endpoints: [
      {
        method: "GET",
        path: "/api/history",
        description: "Get user's activity history",
        responses: {
          "200": "Successful response with history entries and pagination info",
          "401": "Unauthorized",
          "500": "Server error"
        }
      },
      {
        method: "POST",
        path: "/api/history",
        description: "Add a history entry",
        requestBody: `{
  "activity_type": "linkedin_search" | "resume_upload" | "job_suggestion" | "profile_update" | "subscription_change" | "login" | "logout",
  "details": object
}`,
        responses: {
          "200": "Successful response with added history entry",
          "400": "Invalid request parameters",
          "401": "Unauthorized",
          "500": "Server error"
        }
      },
      {
        method: "POST",
        path: "/api/history/clear",
        description: "Clear user's history",
        responses: {
          "200": `{
  "message": string
}`,
          "401": "Unauthorized",
          "500": "Server error"
        }
      },
      {
        method: "GET",
        path: "/api/history/stats",
        description: "Get user's history stats",
        responses: {
          "200": "Successful response with activity counts, recent activity, and total activities",
          "401": "Unauthorized",
          "500": "Server error"
        }
      }
    ]
  },
  {
    name: "Webhook",
    endpoints: [
      {
        method: "POST",
        path: "/api/webhook",
        description: "Stripe webhook",
        responses: {
          "200": `{
  "status": "success"
}`,
          "400": "Invalid payload",
          "500": "Server error"
        }
      }
    ]
  },
  {
    name: "Root",
    endpoints: [
      {
        method: "GET",
        path: "/",
        description: "Root endpoint",
        responses: {
          "200": "Successful response with API info or redirect to frontend URL"
        }
      }
    ]
  }
]

const EndpointCard: React.FC<{ endpoint: Endpoint }> = ({ endpoint }) => (
  <Card>
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Badge variant={endpoint.method === 'GET' ? 'secondary' : 'default'}>
          {endpoint.method}
        </Badge>
        <span className="font-mono text-sm">{endpoint.path}</span>
      </CardTitle>
      <CardDescription>{endpoint.description}</CardDescription>
    </CardHeader>
    <CardContent>
      {endpoint.requestBody && (
        <div className="mb-4">
          <h4 className="text-sm font-semibold mb-2">Request Body:</h4>
          <pre className="bg-gray-100 p-2 rounded-md overflow-x-auto">
            <code>{endpoint.requestBody}</code>
          </pre>
        </div>
      )}
      <h4 className="text-sm font-semibold mb-2">Responses:</h4>
      <ul className="list-disc pl-5">
        {Object.entries(endpoint.responses).map(([code, description]) => (
          <li key={code} className="mb-1">
            <span className="font-semibold">{code}:</span> {description}
          </li>
        ))}
      </ul>
    </CardContent>
  </Card>
)

export const ApiDocumentation: React.FC = () => {
  return (
    <div className="space-y-8">
      <h1 className="text-3xl font-bold mb-6">API Documentation</h1>
      <Accordion type="single" collapsible className="w-full">
        {endpointGroups.map((group, index) => (
          <AccordionItem value={`item-${index}`} key={index}>
            <AccordionTrigger>{group.name}</AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4">
                {group.endpoints.map((endpoint, endpointIndex) => (
                  <EndpointCard key={endpointIndex} endpoint={endpoint} />
                ))}
              </div>
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </div>
  )
}

